package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/smtp"
	"os"
	"strings"
	"time"
)

// ========== Puzzle Monitor ==========

type PuzzleMonitor struct {
	coord   *Coordinator
	emailTo string
}

func NewPuzzleMonitor(coord *Coordinator, emailTo string) *PuzzleMonitor {
	return &PuzzleMonitor{coord: coord, emailTo: emailTo}
}

func (m *PuzzleMonitor) Start() {
	// Send startup test email
	stats := m.coord.getStats()
	m.sendEmail(
		fmt.Sprintf("[Puzzle Pool] Coordinator started - targeting #%d", stats.PuzzleNum),
		fmt.Sprintf("Coordinator is online.\n\nPuzzle: #%d\nAddress: %s\nChunk size: 2^%d\nTotal chunks: %s\nStarted at: %s",
			stats.PuzzleNum, stats.Address, stats.ChunkBits,
			stats.TotalChunks, time.Now().UTC().Format("2006-01-02 15:04:05 UTC")))

	go m.monitorLoop()
	go m.dailyReportLoop()
	log.Printf("[Monitor] Started: check every 1h, email=%s", m.emailTo)
}

// Check if puzzle's address has been emptied (solved by someone)
func (m *PuzzleMonitor) isPuzzleSolved(puzzleNum int) (bool, string) {
	if puzzleNum < 1 || puzzleNum > len(m.coord.puzzles) {
		return false, ""
	}
	puzzle := m.coord.puzzles[puzzleNum-1]
	if puzzle.Key != "" {
		return true, puzzle.Key
	}

	// Query mempool.space API
	url := fmt.Sprintf("https://mempool.space/api/address/%s", puzzle.Addr)
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		log.Printf("[Monitor] API error #%d: %v", puzzleNum, err)
		return false, ""
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	var data struct {
		ChainStats struct {
			FundedSum uint64 `json:"funded_txo_sum"`
			SpentSum  uint64 `json:"spent_txo_sum"`
		} `json:"chain_stats"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return false, ""
	}

	balance := data.ChainStats.FundedSum - data.ChainStats.SpentSum
	if balance == 0 && data.ChainStats.FundedSum > 0 {
		return true, ""
	}
	return false, ""
}

func (m *PuzzleMonitor) findNextUnsolved(startFrom int) int {
	for i := startFrom; i <= 160; i++ {
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

func (m *PuzzleMonitor) monitorLoop() {
	time.Sleep(30 * time.Second) // initial delay

	for {
		current := m.coord.puzzleNum
		solved, key := m.isPuzzleSolved(current)
		if solved {
			log.Printf("[Monitor] Puzzle #%d SOLVED!", current)
			next := m.findNextUnsolved(current + 1)
			if next > 0 {
				log.Printf("[Monitor] Switching to #%d", next)
				m.coord.switchPuzzle(next)
				m.sendEmail(
					fmt.Sprintf("[Puzzle Pool] #%d solved! Now targeting #%d", current, next),
					fmt.Sprintf("Puzzle #%d has been solved (by someone on the network).\nKey: %s\n\nAuto-switched to puzzle #%d (%s).\nWorkers will sync on next task fetch.",
						current, key, next, m.coord.puzzles[next-1].Addr))
			}
		}
		time.Sleep(1 * time.Hour)
	}
}

func (m *PuzzleMonitor) dailyReportLoop() {
	// Align to next 9:00 UTC
	now := time.Now().UTC()
	next := time.Date(now.Year(), now.Month(), now.Day(), 9, 0, 0, 0, time.UTC)
	if now.After(next) {
		next = next.Add(24 * time.Hour)
	}
	time.Sleep(time.Until(next))

	for {
		m.sendDailyReport()
		time.Sleep(24 * time.Hour)
	}
}

func (m *PuzzleMonitor) sendDailyReport() {
	stats := m.coord.getStats()
	uptime := time.Since(m.coord.startTime)

	etaStr := "N/A"
	if stats.Completed > 0 && m.coord.totalChunks.IsUint64() {
		total := m.coord.totalChunks.Uint64()
		rate := float64(stats.Completed) / uptime.Seconds()
		if rate > 0 {
			remaining := total - stats.Completed
			etaSec := float64(remaining) / rate
			if etaSec > 365*24*3600 {
				etaStr = fmt.Sprintf("%.1f years", etaSec/31536000)
			} else if etaSec > 24*3600 {
				etaStr = fmt.Sprintf("%.1f days", etaSec/86400)
			} else {
				etaStr = fmt.Sprintf("%.1f hours", etaSec/3600)
			}
		}
	}

	subject := fmt.Sprintf("[Puzzle Pool] Daily - #%d - %s", stats.PuzzleNum, time.Now().Format("2006-01-02"))
	body := fmt.Sprintf(`=== Bitcoin Puzzle Pool Daily Report ===
Time:            %s
Puzzle:          #%d (%s)
Bit Range:       %d bits

--- Progress ---
Completed:       %d chunks
In-Flight:       %d chunks
Total Chunks:    %s
Keys Searched:   %s
Found Key:       %s

--- Pool ---
Active Workers:  %d
Uptime:          %s
ETA:             %s
`,
		time.Now().UTC().Format("2006-01-02 15:04 UTC"),
		stats.PuzzleNum, stats.Address, stats.BitRange,
		stats.Completed, stats.InFlight, stats.TotalChunks,
		stats.KeysSearched,
		func() string {
			if stats.FoundKey != "" {
				return stats.FoundKey
			}
			return "(none)"
		}(),
		stats.ActiveWorkers, stats.Uptime, etaStr)

	m.sendEmail(subject, body)
}

func (m *PuzzleMonitor) sendEmail(subject, body string) {
	if m.emailTo == "" {
		return
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
		log.Printf("[Monitor] SMTP not configured. Subject: %s", subject)
		return
	}

	msg := fmt.Sprintf("From: %s\r\nTo: %s\r\nSubject: %s\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n%s",
		smtpUser, m.emailTo, subject, body)

	auth := smtp.PlainAuth("", smtpUser, smtpPass, smtpHost)
	err := smtp.SendMail(smtpHost+":"+smtpPort, auth, smtpUser, strings.Split(m.emailTo, ","), []byte(msg))
	if err != nil {
		log.Printf("[Monitor] Email failed: %v", err)
	} else {
		log.Printf("[Monitor] Email sent: %s", subject)
	}
}
