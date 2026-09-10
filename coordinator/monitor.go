package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/smtp"
	"os"
	"strings"
	"time"
)

// ========== HTTP client ==========
//
// The server has no IPv6 connectivity (ping6 -> "Network is unreachable"),
// but mempool.space's DNS returns AAAA records. Go's default client wastes
// time attempting IPv6 before falling back. We force tcp4 to skip that.
// International endpoints are also flaky from this China-based host, so the
// caller retries and falls back to a second endpoint.
var httpClient = &http.Client{
	Timeout: 20 * time.Second,
	Transport: &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			d := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
			return d.DialContext(ctx, "tcp4", addr) // IPv4 only
		},
	},
}

// ========== Puzzle Monitor ==========
//
// Three things are meant to reach the operator's inbox, and each has its own
// guarantee:
//
//   1. key found        -> emailed the moment a worker submits it (foundKeyLoop).
//                          Backstopped by the daily report, which keeps listing
//                          any find whose alert was not delivered.
//   2. daily report     -> fired at a fixed wall-clock time, recomputed every
//                          iteration so a long-running process cannot drift.
//   3. puzzle solved    -> the target address is polled; on a hit the alert is
//                          sent and the pool moves to the next unsolved puzzle.

type PuzzleMonitor struct {
	coord   *Coordinator
	emailTo string

	// solveCheckInterval is how often the target address is polled for having
	// been solved by someone else. reportAtUTC is the time of day (UTC) the
	// daily report fires, expressed as an offset from midnight.
	solveCheckInterval time.Duration
	reportAtUTC        time.Duration
}

func NewPuzzleMonitor(coord *Coordinator, emailTo string, solveCheckInterval time.Duration, reportAtUTC time.Duration) *PuzzleMonitor {
	if solveCheckInterval <= 0 {
		solveCheckInterval = 10 * time.Minute
	}
	return &PuzzleMonitor{
		coord:              coord,
		emailTo:            emailTo,
		solveCheckInterval: solveCheckInterval,
		reportAtUTC:        reportAtUTC,
	}
}

func (m *PuzzleMonitor) Start() {
	stats := m.coord.getStats()
	m.sendEmail(
		fmt.Sprintf("[Puzzle Pool] Coordinator started - targeting #%d", stats.PuzzleNum),
		fmt.Sprintf(`Coordinator is online.

Puzzle:         #%d
Address:        %s
Chunk size:     2^%d
Total chunks:   %s
Started at:     %s

Alerting:
  - a found key is emailed as soon as a worker submits it
  - the daily report is emailed at %s UTC
  - the target address is polled every %s; if it shows up as emptied, the pool
    switches to the next unsolved puzzle automatically and emails you
`,
			stats.PuzzleNum, stats.Address, stats.ChunkBits, stats.TotalChunks,
			time.Now().UTC().Format("2006-01-02 15:04:05 UTC"),
			formatClock(m.reportAtUTC), m.solveCheckInterval))

	go m.foundKeyLoop()
	go m.solveLoop()
	go m.dailyReportLoop()

	log.Printf("[Monitor] Started: found-key alerts on submit, solve check every %s, daily report at %s UTC, email=%s",
		m.solveCheckInterval, formatClock(m.reportAtUTC), m.emailTo)
}

// ========== 1. Key found ==========

// foundKeyLoop drains the coordinator's found-key events and emails each one
// immediately.
//
// This is the only *timely* notification path. Without it a found key would sit
// silently in the database until the next daily report - up to 24 hours later -
// and if the pool auto-switched to another puzzle in between, the report would
// no longer mention it at all.
func (m *PuzzleMonitor) foundKeyLoop() {
	for rec := range m.coord.FoundEvents() {
		m.alertFoundKey(rec)
	}
}

func (m *PuzzleMonitor) alertFoundKey(rec FoundRecord) {
	// A worker whose submit response was lost retries the batch, so the same
	// find can be reported twice. Only alert once per (puzzle, key).
	if m.coord.foundKeyNotified(rec.PuzzleNum, rec.KeyHex) {
		return
	}

	subject := fmt.Sprintf("[Puzzle Pool] KEY FOUND - puzzle #%d", rec.PuzzleNum)
	body := fmt.Sprintf(`A worker has submitted the private key for puzzle #%d.

Key:           %s
Address:       %s
Worker:        %s
Chunk index:   %d
Submitted at:  %s

This is only the real solution if the key reproduces the address: derive the
compressed public key from the key, take hash160(SHA256(pubkey)), and compare it
with the hash160 of the address above.

The pool keeps working on the same puzzle until its address shows up as emptied
on-chain; the monitor then switches to the next unsolved puzzle automatically.
`,
		rec.PuzzleNum, rec.KeyHex, rec.Address, rec.WorkerID, rec.ChunkIndex,
		time.Now().UTC().Format("2006-01-02 15:04:05 UTC"))

	if m.sendEmail(subject, body) {
		m.coord.markFoundKeyNotified(rec.PuzzleNum, rec.KeyHex)
		log.Printf("[Monitor] found-key alert sent (puzzle #%d)", rec.PuzzleNum)
		return
	}
	// Leave notified=0 so the daily report repeats it until an email gets out.
	log.Printf("[Monitor] found-key alert FAILED (puzzle #%d); the daily report will keep retrying", rec.PuzzleNum)
}

// ========== 2. Daily report ==========

// dailyReportLoop sends the report at a fixed UTC wall-clock time every day.
//
// The next fire time is recomputed from the wall clock on every iteration
// instead of sleeping a flat 24h: that way a long-running process cannot
// accumulate drift, and a clock step (NTP, manual change) re-aligns instead of
// shifting the report time forever.
func (m *PuzzleMonitor) dailyReportLoop() {
	for {
		now := time.Now().UTC()
		next := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC).Add(m.reportAtUTC)
		if !now.Before(next) {
			next = next.Add(24 * time.Hour)
		}
		log.Printf("[Monitor] next daily report: %s UTC (in %s)",
			next.Format("2006-01-02 15:04"), time.Until(next).Round(time.Second))
		time.Sleep(time.Until(next))
		m.sendDailyReport()
	}
}

func (m *PuzzleMonitor) sendDailyReport() {
	stats := m.coord.getStats()

	// The rate is measured over the time spent on the *current* puzzle. After an
	// automatic switch `completed` resets to 0 while the process keeps running,
	// so dividing by the process uptime would understate the rate and inflate
	// the ETA.
	puzzleUptime := time.Since(m.coord.puzzleStartedAt())
	etaStr := "N/A"
	if stats.Completed > 0 && puzzleUptime > 0 && m.coord.totalChunks.IsUint64() {
		total := m.coord.totalChunks.Uint64()
		if rate := float64(stats.Completed) / puzzleUptime.Seconds(); rate > 0 && total > stats.Completed {
			etaStr = humanDuration(float64(total-stats.Completed) / rate)
		}
	}

	foundKey := stats.FoundKey
	if foundKey == "" {
		foundKey = "(none)"
	}

	subject := fmt.Sprintf("[Puzzle Pool] Daily - #%d - %s", stats.PuzzleNum, time.Now().Format("2006-01-02"))
	body := fmt.Sprintf(`=== Bitcoin Puzzle Pool Daily Report ===
Time:             %s
Puzzle:           #%d (%s)
Bit Range:        %d bits

--- Progress ---
Completed:        %d chunks
In-Flight:        %d chunks
Total Chunks:     %s
Keys Searched:    %s
Found Key:        %s

--- Pool ---
Active Workers:   %d
Time On Puzzle:   %s
Process Uptime:   %s
ETA:              %s
`,
		time.Now().UTC().Format("2006-01-02 15:04 UTC"),
		stats.PuzzleNum, stats.Address, stats.BitRange,
		stats.Completed, stats.InFlight, stats.TotalChunks,
		stats.KeysSearched, foundKey,
		stats.ActiveWorkers, puzzleUptime.Round(time.Second), stats.Uptime, etaStr)

	// Backstop for the immediate alert. A find whose alert email never got
	// through (SMTP down, expired auth code) would otherwise sit unseen in the
	// database, and once the pool auto-switches the "Found Key" line above moves
	// on to the next puzzle. This section is what guarantees a key found by the
	// pool reaches the operator by email at least once.
	pending := m.coord.unnotifiedFoundKeys()
	if len(pending) > 0 {
		body += "\n--- FOUND KEYS WHOSE ALERT EMAIL DID NOT GET THROUGH ---\n"
		for _, p := range pending {
			body += fmt.Sprintf("  puzzle #%d  key=%s  worker=%s  found_at=%s\n",
				p.PuzzleNum, p.KeyHex, p.WorkerID, p.FoundAt)
		}
		body += "\nThese are repeated on every report until an email is delivered.\n"
	}

	if !m.sendEmail(subject, body) {
		return // keep notified=0 so the next report retries
	}
	for _, p := range pending {
		m.coord.markFoundKeyNotified(p.PuzzleNum, p.KeyHex)
	}
}

// ========== 3. Puzzle solved + auto switch ==========

// solveLoop polls the target address and, once it has been emptied, alerts the
// operator and moves the pool to the next unsolved puzzle.
func (m *PuzzleMonitor) solveLoop() {
	time.Sleep(30 * time.Second) // let the HTTP listener come up first

	lastAlerted := 0 // puzzle number already reported as solved
	retryTarget := 0 // switch target left over from a failed attempt

	for {
		current := m.coord.puzzleNum
		solved, remoteKey := m.isPuzzleSolved(current)

		if solved {
			if lastAlerted != current {
				lastAlerted = current
				log.Printf("[Monitor] Puzzle #%d SOLVED!", current)

				retryTarget = m.findNextUnsolved(current + 1) // may be -1
				keyLine := m.keyLineFor(current, remoteKey)

				switch {
				case retryTarget <= 0:
					log.Printf("[Monitor] No unsolved puzzle after #%d", current)
					m.sendEmail(
						fmt.Sprintf("[Puzzle Pool] #%d solved - no unsolved puzzle left", current),
						fmt.Sprintf(`Puzzle #%d has been solved, but there is no unsolved puzzle after it.

Key: %s

The coordinator is still targeting #%d and no further automatic switch is
possible. Point it at a new target manually (PUZZLE_NUM=... ) and restart.
`, current, keyLine, current))

				default:
					if err := m.coord.switchPuzzle(retryTarget); err != nil {
						log.Printf("[Monitor] Switch #%d -> #%d failed: %v", current, retryTarget, err)
						m.sendEmail(
							fmt.Sprintf("[Puzzle Pool] #%d solved - switch to #%d FAILED", current, retryTarget),
							fmt.Sprintf(`Puzzle #%d has been solved, but switching to #%d failed:

%v

The coordinator is still targeting #%d. The switch is retried every %s and you
will get another email if it succeeds.
`, current, retryTarget, err, current, m.solveCheckInterval))
						// Keep retryTarget set: the branch below retries silently.
					} else {
						log.Printf("[Monitor] Switched to #%d", retryTarget)
						m.sendEmail(
							fmt.Sprintf("[Puzzle Pool] #%d solved - now targeting #%d", current, retryTarget),
							fmt.Sprintf(`Puzzle #%d has been solved.

Key: %s

Auto-switched to puzzle #%d (%s).
Workers pick up the new target on their next task fetch.
`, current, keyLine, retryTarget, m.coord.puzzles[retryTarget-1].Addr))
						retryTarget = 0
					}
				}
			} else if retryTarget > 0 {
				// Silent retry of a switch that failed earlier; only email on success.
				if err := m.coord.switchPuzzle(retryTarget); err == nil {
					log.Printf("[Monitor] Retry succeeded, switched to #%d", retryTarget)
					m.sendEmail(
						fmt.Sprintf("[Puzzle Pool] now targeting #%d", retryTarget),
						fmt.Sprintf("The delayed switch succeeded: the pool is now targeting puzzle #%d (%s).\n",
							retryTarget, m.coord.puzzles[retryTarget-1].Addr))
					retryTarget = 0
				} else {
					log.Printf("[Monitor] Retry switch to #%d failed: %v", retryTarget, err)
				}
			}
		}

		time.Sleep(m.solveCheckInterval)
	}
}

// keyLineFor describes whose key solved the puzzle, preferring the one this
// pool actually has on record.
func (m *PuzzleMonitor) keyLineFor(puzzleNum int, remoteKey string) string {
	if ours := m.coord.foundKeyFor(puzzleNum); ours != "" {
		return ours + "   <-- found by THIS pool"
	}
	if remoteKey != "" {
		return remoteKey + "   <-- from puzzles.json"
	}
	return "(unknown - solved by someone else on the network)"
}

// ========== Chain queries ==========

// isPuzzleSolved reports whether a puzzle's address has been emptied.
// Tries multiple endpoints (mempool.space, then blockstream.info) with
// retries, since international links from this host are flaky.
//
// A network failure returns false, i.e. "not solved": never announce a solve we
// could not confirm.
func (m *PuzzleMonitor) isPuzzleSolved(puzzleNum int) (bool, string) {
	if puzzleNum < 1 || puzzleNum > len(m.coord.puzzles) {
		return false, ""
	}
	puzzle := m.coord.puzzles[puzzleNum-1]
	if puzzle.Key != "" {
		return true, puzzle.Key
	}

	// Both endpoints use the same mempool.space API shape.
	endpoints := []string{
		fmt.Sprintf("https://mempool.space/api/address/%s", puzzle.Addr),
		fmt.Sprintf("https://blockstream.info/api/address/%s", puzzle.Addr),
	}

	for _, url := range endpoints {
		for attempt := 1; attempt <= 2; attempt++ {
			balance, funded, ok := m.queryBalance(url)
			if ok {
				if balance == 0 && funded > 0 {
					return true, ""
				}
				return false, ""
			}
			if attempt < 2 {
				time.Sleep(2 * time.Second)
			}
		}
	}
	return false, ""
}

// queryBalance fetches funded/spent sums from one mempool.space-compatible
// endpoint. Returns ok=false on network/HTTP/parse errors.
func (m *PuzzleMonitor) queryBalance(url string) (balance, funded uint64, ok bool) {
	resp, err := httpClient.Get(url)
	if err != nil {
		log.Printf("[Monitor] API error: %s: %v", url, err)
		return 0, 0, false
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		log.Printf("[Monitor] API error: %s: HTTP %d", url, resp.StatusCode)
		return 0, 0, false
	}
	var data struct {
		ChainStats struct {
			FundedSum uint64 `json:"funded_txo_sum"`
			SpentSum  uint64 `json:"spent_txo_sum"`
		} `json:"chain_stats"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		log.Printf("[Monitor] API error: %s: parse: %v", url, err)
		return 0, 0, false
	}
	return data.ChainStats.FundedSum - data.ChainStats.SpentSum, data.ChainStats.FundedSum, true
}

func (m *PuzzleMonitor) findNextUnsolved(startFrom int) int {
	// Bound by the loaded puzzle list, not a hardcoded 160: indexing past the
	// end would panic the whole coordinator.
	for i := startFrom; i <= len(m.coord.puzzles); i++ {
		if m.coord.puzzles[i-1].Key != "" {
			continue // already known solved
		}
		solved, _ := m.isPuzzleSolved(i)
		if !solved {
			return i
		}
		time.Sleep(2 * time.Second) // rate limit
	}
	return -1
}

// ========== Helpers ==========

func formatClock(offset time.Duration) string {
	return fmt.Sprintf("%02d:%02d", int(offset/time.Hour), int((offset%time.Hour)/time.Minute))
}

// parseClock reads an "HH:MM" wall-clock time as an offset from midnight.
func parseClock(v string) (time.Duration, error) {
	var h, m int
	if _, err := fmt.Sscanf(v, "%d:%d", &h, &m); err != nil {
		return 0, fmt.Errorf("expected HH:MM, got %q", v)
	}
	if h < 0 || h > 23 || m < 0 || m > 59 {
		return 0, fmt.Errorf("time out of range: %q", v)
	}
	return time.Duration(h)*time.Hour + time.Duration(m)*time.Minute, nil
}

func humanDuration(seconds float64) string {
	switch {
	case seconds > 365*24*3600:
		return fmt.Sprintf("%.1f years", seconds/31536000)
	case seconds > 24*3600:
		return fmt.Sprintf("%.1f days", seconds/86400)
	default:
		return fmt.Sprintf("%.1f hours", seconds/3600)
	}
}

// sendEmail delivers one message. It reports whether the message actually left
// the process, so callers can decide whether to retry later - a silent failure
// here is exactly what makes an alert disappear.
func (m *PuzzleMonitor) sendEmail(subject, body string) bool {
	if m.emailTo == "" {
		log.Printf("[Monitor] EMAIL_TO is empty, not sending: %s", subject)
		return false
	}
	smtpHost := os.Getenv("SMTP_HOST")
	smtpPort := os.Getenv("SMTP_PORT")
	smtpUser := os.Getenv("SMTP_USER")
	smtpPass := os.Getenv("SMTP_PASS")

	if smtpHost == "" {
		smtpHost = "smtp.qq.com"
	}
	if smtpPort == "" {
		smtpPort = "587"
	}

	if smtpUser == "" || smtpPass == "" {
		log.Printf("[Monitor] SMTP not configured, not sending: %s", subject)
		return false
	}

	msg := fmt.Sprintf("From: %s\r\nTo: %s\r\nSubject: %s\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n%s",
		smtpUser, m.emailTo, subject, body)

	auth := smtp.PlainAuth("", smtpUser, smtpPass, smtpHost)
	if err := smtp.SendMail(smtpHost+":"+smtpPort, auth, smtpUser, strings.Split(m.emailTo, ","), []byte(msg)); err != nil {
		log.Printf("[Monitor] Email FAILED: %s: %v", subject, err)
		return false
	}
	log.Printf("[Monitor] Email sent: %s", subject)
	return true
}
