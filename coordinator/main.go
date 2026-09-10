package main

import (
	"crypto/rand"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// ========== Puzzle Data ==========

type PuzzleEntry struct {
	N    int    `json:"n"`
	Addr string `json:"addr"`
	Key  string `json:"key"`
}

// ========== Task Allocation: In-Memory Random Queue ==========
//
// Problem: If we store all completed chunks in SQLite, puzzle #71 with
// chunk_bits=30 has 2^40 = 1 trillion possible chunks. Storing each
// completed chunk as a row would require 16 TB of SQLite.
//
// Solution: "Lazy Shuffle" approach.
//   - We DON'T pre-generate or store all chunk indices.
//   - We maintain a set of "issued" chunk indices (in-flight + completed).
//   - On each request, we generate a random index and check membership.
//   - Issued set is stored as a bitmap in SQLite, grouped into pages.
//   - For 2^40 chunks at chunk_bits=30: bitmap = 2^40 bits = 128 GB.
//     STILL TOO BIG for full bitmap.
//
// Better Solution: "Reservoir with DB dedup"
//   - Keep only the ISSUED indices in DB (not all possible ones).
//   - At any time, issued = assigned + completed (typically tiny vs total).
//   - Random selection: pick random index, check if in issued set.
//   - When completion rate is low (99.9999% of the time for puzzle #71),
//     collision probability is negligible (P(collision) = issued/total).
//   - Even at 1 year of 100 workers @ 10MK/s: ~3M tasks completed.
//     3M / 2^40 ≈ 0.0003% collision rate. Random retry is instant.
//   - When completion rate gets HIGH (>50%), we switch to sequential scan
//     of the remaining range. But this won't happen in our lifetimes for #71.
//
// This design: O(1) allocation, O(issued_count) storage, collision-free in practice.

// ========== Models ==========

type Task struct {
	ID         int64  `json:"id"`
	ChunkIndex uint64 `json:"chunk_index"`
	StartHex   string `json:"start_hex"`
	Size       uint64 `json:"size"`
	Status     int    `json:"status"` // 1=assigned
	WorkerID   string `json:"worker_id,omitempty"`
	AssignedAt int64  `json:"assigned_at,omitempty"` // unix timestamp
}

type GetBatchResp struct {
	Tasks      []Task `json:"tasks"`
	TargetH160 string `json:"target_h160"`
	PuzzleNum  int    `json:"puzzle_num"`
	BitRange   int    `json:"bit_range"`
	Error      string `json:"error,omitempty"`
}

type SubmitReq struct {
	WorkerID string         `json:"worker_id"`
	Results  []SubmitResult `json:"results"`
}

type SubmitResult struct {
	TaskID int64  `json:"task_id"`
	Found  bool   `json:"found"`
	KeyHex string `json:"key_hex,omitempty"`
}

// FoundRecord is one solved puzzle, handed to the monitor so it can alert the
// operator immediately instead of waiting for the daily report.
type FoundRecord struct {
	PuzzleNum  int
	KeyHex     string
	WorkerID   string
	ChunkIndex uint64
	Address    string
}

// PendingFound is a recorded find whose alert email has not been confirmed yet.
type PendingFound struct {
	PuzzleNum int
	KeyHex    string
	WorkerID  string
	FoundAt   string
}

type StatsResp struct {
	PuzzleNum     int    `json:"puzzle_num"`
	Address       string `json:"address"`
	BitRange      int    `json:"bit_range"`
	ChunkBits     int    `json:"chunk_bits"`
	TotalChunks   string `json:"total_chunks"` // string because can be huge
	Completed     uint64 `json:"completed"`
	InFlight      int    `json:"in_flight"`
	ActiveWorkers int    `json:"active_workers"`
	KeysSearched  string `json:"keys_searched"`
	FoundKey      string `json:"found_key,omitempty"`
	Uptime        string `json:"uptime"`
}

// ========== Coordinator ==========

type Coordinator struct {
	db          *sql.DB
	puzzles     []PuzzleEntry
	puzzleNum   int
	chunkBits   int
	chunkSize   uint64
	totalChunks *big.Int // can be very large
	rangeStart  *big.Int
	targetH160  string
	startTime   time.Time

	// puzzleStartTime is when the coordinator started working on the *current*
	// puzzle. It resets on switchPuzzle, and is the correct baseline for the
	// throughput/ETA in the daily report (process uptime is not).
	puzzleStartTime time.Time

	// foundCh carries found keys to the monitor. Buffered and written to
	// non-blockingly: if the monitor is busy the record is simply not queued,
	// because the row stays notified=0 in the database and the daily report
	// picks it up as a backstop.
	foundCh chan FoundRecord

	// Concurrency: mutex protects the in-memory allocation state
	mu      sync.Mutex
	workers map[string]time.Time // worker_id -> last_seen

	// Atomic counters (avoid COUNT(*) scans)
	completedCount uint64

	// Prepared statements for hot paths
	stmtInsertTask  *sql.Stmt
	stmtDeleteTask  *sql.Stmt
	stmtInsertDone  *sql.Stmt
	stmtCheckDone   *sql.Stmt
	stmtCheckActive *sql.Stmt
}

func NewCoordinator(dbPath string, puzzles []PuzzleEntry, puzzleNum, chunkBits int) (*Coordinator, error) {
	db, err := sql.Open("sqlite3", dbPath+"?_journal=WAL&_sync=NORMAL&_busy_timeout=5000&cache=shared")
	if err != nil {
		return nil, err
	}
	// Connection pool: SQLite supports 1 writer but multiple readers.
	// Set to 2 to avoid deadlock when Query rows are open during Exec.
	db.SetMaxOpenConns(2)
	db.SetMaxIdleConns(2)

	c := &Coordinator{
		db:              db,
		puzzles:         puzzles,
		puzzleNum:       puzzleNum,
		chunkBits:       chunkBits,
		workers:         make(map[string]time.Time),
		startTime:       time.Now(),
		puzzleStartTime: time.Now(),
		foundCh:         make(chan FoundRecord, 64),
	}

	c.chunkSize = uint64(1) << uint(chunkBits)
	c.rangeStart = new(big.Int).Lsh(big.NewInt(1), uint(puzzleNum-1))
	// totalChunks = 2^(puzzleNum-1) / 2^chunkBits = 2^(puzzleNum-1-chunkBits)
	exp := puzzleNum - 1 - chunkBits
	if exp < 0 {
		exp = 0
	}
	c.totalChunks = new(big.Int).Lsh(big.NewInt(1), uint(exp))

	puzzle := puzzles[puzzleNum-1]
	h160, err := addressToH160Hex(puzzle.Addr)
	if err != nil {
		// Fail loudly rather than mining a target that can never be found.
		return nil, fmt.Errorf("puzzle #%d: %w", puzzleNum, err)
	}
	c.targetH160 = h160

	if err := c.initDB(); err != nil {
		return nil, err
	}
	if err := c.prepareStmts(); err != nil {
		return nil, err
	}

	// Load completed count from DB at startup
	c.db.QueryRow(`SELECT COUNT(*) FROM done`).Scan(&c.completedCount)

	go c.reapLoop()
	return c, nil
}

func (c *Coordinator) initDB() error {
	_, err := c.db.Exec(`
		-- Active tasks: only in-flight tasks live here.
		-- Deleted on completion. Typically < 1000 rows.
		CREATE TABLE IF NOT EXISTS active_tasks (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			chunk_index INTEGER NOT NULL,
			start_hex TEXT NOT NULL,
			size INTEGER NOT NULL,
			worker_id TEXT NOT NULL,
			assigned_at INTEGER NOT NULL
		);
		CREATE UNIQUE INDEX IF NOT EXISTS idx_active_chunk ON active_tasks(chunk_index);

		-- Completed chunks: only stores the chunk_index.
		-- 8 bytes per entry. 1M completions = 8MB. Manageable for years.
		CREATE TABLE IF NOT EXISTS done (
			chunk_index INTEGER PRIMARY KEY
		) WITHOUT ROWID;

		CREATE TABLE IF NOT EXISTS found_keys (
			puzzle_num INTEGER PRIMARY KEY,
			key_hex TEXT NOT NULL,
			worker_id TEXT,
			found_at TEXT DEFAULT (datetime('now'))
		);
	`)
	if err != nil {
		return err
	}

	// Migration for databases created before alert tracking existed. ADD COLUMN
	// errors with "duplicate column name" once the column is present, which is
	// the normal case on every start after the first - only that error is
	// tolerated.
	if _, err := c.db.Exec(`ALTER TABLE found_keys ADD COLUMN notified INTEGER NOT NULL DEFAULT 0`); err != nil &&
		!strings.Contains(err.Error(), "duplicate column name") {
		return err
	}
	return nil
}

func (c *Coordinator) prepareStmts() error {
	var err error
	c.stmtInsertTask, err = c.db.Prepare(
		`INSERT OR IGNORE INTO active_tasks (chunk_index, start_hex, size, worker_id, assigned_at) VALUES (?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	c.stmtDeleteTask, err = c.db.Prepare(`DELETE FROM active_tasks WHERE id = ?`)
	if err != nil {
		return err
	}
	c.stmtInsertDone, err = c.db.Prepare(`INSERT OR IGNORE INTO done (chunk_index) VALUES (?)`)
	if err != nil {
		return err
	}
	c.stmtCheckDone, err = c.db.Prepare(`SELECT 1 FROM done WHERE chunk_index = ?`)
	if err != nil {
		return err
	}
	c.stmtCheckActive, err = c.db.Prepare(`SELECT 1 FROM active_tasks WHERE chunk_index = ?`)
	return err
}

// isChunkAvailable checks if a chunk index is neither done nor in-flight.
func (c *Coordinator) isChunkAvailable(idx uint64) bool {
	var x int
	if c.stmtCheckDone.QueryRow(idx).Scan(&x) == nil {
		return false // already completed
	}
	if c.stmtCheckActive.QueryRow(idx).Scan(&x) == nil {
		return false // currently assigned
	}
	return true
}

// allocateChunks picks N random available chunk indices and assigns them.
// Priority: 1) reclaim stale tasks first, 2) random new chunks.
// Holds the mutex for the entire operation to guarantee no duplicates.
func (c *Coordinator) allocateChunks(workerID string, count int) ([]Task, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now().Unix()
	c.workers[workerID] = time.Now()

	var tasks []Task

	// Priority 1: reclaim stale tasks (worker died without completing)
	// IMPORTANT: Read all rows first, then close, then update.
	// With MaxOpenConns=1, an open Rows blocks all other SQL operations.
	type staleRow struct {
		id         int64
		chunkIndex uint64
		startHex   string
		size       uint64
	}
	var staleList []staleRow
	staleRows, err := c.db.Query(
		`SELECT id, chunk_index, start_hex, size FROM active_tasks WHERE worker_id='__stale__' LIMIT ?`, count)
	if err == nil {
		for staleRows.Next() {
			var s staleRow
			staleRows.Scan(&s.id, &s.chunkIndex, &s.startHex, &s.size)
			staleList = append(staleList, s)
		}
		staleRows.Close() // Release connection BEFORE doing updates
	}

	for _, s := range staleList {
		c.db.Exec(`UPDATE active_tasks SET worker_id=?, assigned_at=? WHERE id=?`, workerID, now, s.id)
		tasks = append(tasks, Task{
			ID: s.id, ChunkIndex: s.chunkIndex, StartHex: s.startHex,
			Size: s.size, Status: 1, WorkerID: workerID, AssignedAt: now,
		})
	}

	if len(tasks) >= count {
		return tasks, nil
	}

	// Priority 2: random new chunks
	totalU64 := c.totalChunks.Uint64()
	useBigRand := !c.totalChunks.IsUint64()
	remaining := count - len(tasks)
	maxAttempts := remaining * 50

	for i := 0; i < maxAttempts && len(tasks) < count; i++ {
		var idx uint64
		if useBigRand {
			b := make([]byte, (c.totalChunks.BitLen()+7)/8)
			rand.Read(b)
			n := new(big.Int).SetBytes(b)
			n.Mod(n, c.totalChunks)
			idx = n.Uint64()
		} else {
			buf := make([]byte, 8)
			rand.Read(buf)
			idx = binary.LittleEndian.Uint64(buf) % totalU64
		}

		if !c.isChunkAvailable(idx) {
			continue
		}

		offset := new(big.Int).SetUint64(idx)
		offset.Mul(offset, new(big.Int).SetUint64(c.chunkSize))
		startKey := new(big.Int).Add(c.rangeStart, offset)
		startHex := fmt.Sprintf("%x", startKey)

		res, err := c.stmtInsertTask.Exec(idx, startHex, c.chunkSize, workerID, now)
		if err != nil {
			continue
		}
		rows, _ := res.RowsAffected()
		if rows == 0 {
			continue
		}
		id, _ := res.LastInsertId()
		tasks = append(tasks, Task{
			ID: id, ChunkIndex: idx, StartHex: startHex,
			Size: c.chunkSize, Status: 1, WorkerID: workerID, AssignedAt: now,
		})
	}

	if len(tasks) == 0 {
		return nil, fmt.Errorf("no tasks available")
	}
	return tasks, nil
}

func (c *Coordinator) completeTasks(workerID string, results []SubmitResult) (int, []FoundRecord) {
	c.mu.Lock()

	c.workers[workerID] = time.Now()
	completed := 0
	var found []FoundRecord

	for _, r := range results {
		// Get chunk_index from active task
		var chunkIndex uint64
		err := c.db.QueryRow(`SELECT chunk_index FROM active_tasks WHERE id=?`, r.TaskID).Scan(&chunkIndex)
		if err != nil {
			continue
		}
		// Mark done
		c.stmtInsertDone.Exec(chunkIndex)
		c.stmtDeleteTask.Exec(r.TaskID)
		c.completedCount++
		completed++

		if r.Found && r.KeyHex != "" {
			puzzleNum := c.puzzleNum
			// notified stays 0 until the alert email is confirmed sent. The CASE
			// keeps it 0 when a different key arrives for the same puzzle, but
			// preserves it when a worker replays the same find (its submit
			// response was lost, so the submit thread retried the batch).
			c.db.Exec(`INSERT INTO found_keys (puzzle_num, key_hex, worker_id, notified)
					VALUES (?, ?, ?, 0)
					ON CONFLICT(puzzle_num) DO UPDATE SET
						key_hex   = excluded.key_hex,
						worker_id = excluded.worker_id,
						notified  = CASE WHEN found_keys.key_hex = excluded.key_hex
						                 THEN found_keys.notified ELSE 0 END`,
				puzzleNum, r.KeyHex, workerID)
			log.Printf("!!! KEY FOUND: puzzle #%d key=%s worker=%s chunk=%d",
				puzzleNum, r.KeyHex, workerID, chunkIndex)
			found = append(found, FoundRecord{
				PuzzleNum:  puzzleNum,
				KeyHex:     r.KeyHex,
				WorkerID:   workerID,
				ChunkIndex: chunkIndex,
				Address:    c.puzzles[puzzleNum-1].Addr,
			})
		}
	}

	c.mu.Unlock()

	// Emitted after the lock is released: the monitor's handler writes to the
	// database too, and an SMTP round trip must never run under the allocation
	// mutex (it would stall every worker).
	for _, rec := range found {
		select {
		case c.foundCh <- rec:
		default:
			log.Printf("[Coordinator] found-key queue full; the daily report will still carry puzzle #%d", rec.PuzzleNum)
		}
	}

	return completed, found
}

func (c *Coordinator) reapLoop() {
	for range time.NewTicker(60 * time.Second).C {
		cutoff := time.Now().Add(-10 * time.Minute).Unix()
		c.mu.Lock()
		c.db.Exec(`UPDATE active_tasks SET worker_id='__stale__', assigned_at=0 WHERE assigned_at > 0 AND assigned_at < ?`, cutoff)
		c.mu.Unlock()
	}
}

// Switch to a new puzzle target. Called by the monitor once the current puzzle
// is solved. Workers pick up the new target on their next task fetch.
//
// The new address is validated before any state changes, so a malformed entry in
// puzzles.json leaves the coordinator mining the old target instead of silently
// pointing every worker at a hash160 that can never be hit.
func (c *Coordinator) switchPuzzle(newPuzzleNum int) error {
	if newPuzzleNum < 1 || newPuzzleNum > len(c.puzzles) {
		return fmt.Errorf("puzzle #%d out of range (1..%d)", newPuzzleNum, len(c.puzzles))
	}
	targetH160, err := addressToH160Hex(c.puzzles[newPuzzleNum-1].Addr)
	if err != nil {
		return fmt.Errorf("puzzle #%d: %w", newPuzzleNum, err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Clear in-flight tasks (they're for the old puzzle)
	c.db.Exec(`DELETE FROM active_tasks`)

	// Update coordinator state
	c.puzzleNum = newPuzzleNum
	c.rangeStart = new(big.Int).Lsh(big.NewInt(1), uint(newPuzzleNum-1))

	chunkBits := newPuzzleNum - 1
	if chunkBits > 30 {
		chunkBits = 30
	}
	c.chunkBits = chunkBits
	c.chunkSize = uint64(1) << uint(chunkBits)
	exp := newPuzzleNum - 1 - chunkBits
	if exp < 0 {
		exp = 0
	}
	c.totalChunks = new(big.Int).Lsh(big.NewInt(1), uint(exp))
	c.targetH160 = targetH160
	c.completedCount = 0
	c.puzzleStartTime = time.Now()

	// Reset done table for the new puzzle (old data is no longer relevant).
	// found_keys is deliberately kept: rows for earlier puzzles stay, and any
	// that were never alerted are still reported by the daily backstop.
	c.db.Exec(`DELETE FROM done`)

	log.Printf("[Switch] Now targeting puzzle #%d (%s) h160=%s",
		newPuzzleNum, c.puzzles[newPuzzleNum-1].Addr, targetH160)
	return nil
}

// ========== Found-key plumbing (consumed by the monitor) ==========

// FoundEvents is the stream of fresh finds that the monitor alerts on.
func (c *Coordinator) FoundEvents() <-chan FoundRecord { return c.foundCh }

func (c *Coordinator) puzzleStartedAt() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.puzzleStartTime
}

func (c *Coordinator) foundKeyFor(puzzleNum int) string {
	var key string
	c.db.QueryRow(`SELECT key_hex FROM found_keys WHERE puzzle_num=?`, puzzleNum).Scan(&key)
	return key
}

// foundKeyNotified reports whether the alert for this exact find has already
// been confirmed as delivered.
func (c *Coordinator) foundKeyNotified(puzzleNum int, keyHex string) bool {
	var notified int
	if err := c.db.QueryRow(`SELECT notified FROM found_keys WHERE puzzle_num=? AND key_hex=?`,
		puzzleNum, keyHex).Scan(&notified); err != nil {
		return false
	}
	return notified != 0
}

func (c *Coordinator) markFoundKeyNotified(puzzleNum int, keyHex string) {
	if _, err := c.db.Exec(`UPDATE found_keys SET notified=1 WHERE puzzle_num=? AND key_hex=?`,
		puzzleNum, keyHex); err != nil {
		log.Printf("[Coordinator] Could not mark found key as notified: %v", err)
	}
}

// unnotifiedFoundKeys lists every find whose alert has not been confirmed,
// across all puzzles.
//
// Deliberately not filtered by the current puzzle: after an automatic switch
// that filter would hide a key found moments before the switch, which is
// exactly the case where the operator most needs to be told.
func (c *Coordinator) unnotifiedFoundKeys() []PendingFound {
	rows, err := c.db.Query(
		`SELECT puzzle_num, key_hex, COALESCE(worker_id,''), COALESCE(found_at,'')
		   FROM found_keys WHERE notified=0 ORDER BY puzzle_num`)
	if err != nil {
		log.Printf("[Coordinator] Could not read pending found keys: %v", err)
		return nil
	}
	defer rows.Close()

	var out []PendingFound
	for rows.Next() {
		var p PendingFound
		if err := rows.Scan(&p.PuzzleNum, &p.KeyHex, &p.WorkerID, &p.FoundAt); err == nil {
			out = append(out, p)
		}
	}
	return out
}

func (c *Coordinator) getStats() StatsResp {
	var inFlight int
	c.db.QueryRow(`SELECT COUNT(*) FROM active_tasks`).Scan(&inFlight)

	c.mu.Lock()
	completed := c.completedCount
	active := 0
	cutoff := time.Now().Add(-2 * time.Minute)
	for _, t := range c.workers {
		if t.After(cutoff) {
			active++
		}
	}
	c.mu.Unlock()

	var foundKey string
	c.db.QueryRow(`SELECT key_hex FROM found_keys WHERE puzzle_num=?`, c.puzzleNum).Scan(&foundKey)

	keysSearched := new(big.Int).SetUint64(completed)
	keysSearched.Mul(keysSearched, new(big.Int).SetUint64(c.chunkSize))

	return StatsResp{
		PuzzleNum:     c.puzzleNum,
		Address:       c.puzzles[c.puzzleNum-1].Addr,
		BitRange:      c.puzzleNum,
		ChunkBits:     c.chunkBits,
		TotalChunks:   c.totalChunks.String(),
		Completed:     completed,
		InFlight:      inFlight,
		ActiveWorkers: active,
		KeysSearched:  keysSearched.String(),
		FoundKey:      foundKey,
		Uptime:        time.Since(c.startTime).Round(time.Second).String(),
	}
}

// ========== HTTP Handlers ==========

func (c *Coordinator) handleRegister(w http.ResponseWriter, r *http.Request) {
	var req struct {
		WorkerID string `json:"worker_id"`
		Backend  string `json:"backend"`
		Hostname string `json:"hostname"`
		Rate     uint64 `json:"rate"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	c.mu.Lock()
	c.workers[req.WorkerID] = time.Now()
	c.mu.Unlock()
	log.Printf("Worker registered: %s (%s @ %s, %d keys/s)", req.WorkerID, req.Backend, req.Hostname, req.Rate)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (c *Coordinator) handleGetTasks(w http.ResponseWriter, r *http.Request) {
	workerID := r.URL.Query().Get("worker_id")
	if workerID == "" {
		json.NewEncoder(w).Encode(GetBatchResp{Error: "missing worker_id"})
		return
	}
	count := 1
	fmt.Sscanf(r.URL.Query().Get("count"), "%d", &count)
	if count < 1 {
		count = 1
	}
	if count > 100 {
		count = 100
	}

	tasks, err := c.allocateChunks(workerID, count)
	if err != nil {
		json.NewEncoder(w).Encode(GetBatchResp{Error: err.Error()})
		return
	}
	json.NewEncoder(w).Encode(GetBatchResp{
		Tasks:      tasks,
		TargetH160: c.targetH160,
		PuzzleNum:  c.puzzleNum,
		BitRange:   c.puzzleNum,
	})
}

func (c *Coordinator) handleSubmit(w http.ResponseWriter, r *http.Request) {
	var req SubmitReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", 400)
		return
	}
	completed, found := c.completeTasks(req.WorkerID, req.Results)
	resp := map[string]interface{}{"status": "ok", "completed": completed}
	if len(found) > 0 {
		resp["found_key"] = found[0].KeyHex
	}
	json.NewEncoder(w).Encode(resp)
}

func (c *Coordinator) handleStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(c.getStats())
}

// ========== Helpers ==========

// Base58 alphabet used by Bitcoin legacy addresses.
const base58Alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

// addressToH160Hex decodes a legacy P2PKH address into its 20-byte hash160.
//
// The input is validated rather than decoded optimistically: a single typo in
// puzzles.json would otherwise yield a plausible-looking target that no worker
// can ever hit, and the pool would appear to run normally forever.
func addressToH160Hex(addr string) (string, error) {
	if len(addr) < 26 || len(addr) > 35 {
		return "", fmt.Errorf("address %q has implausible length %d", addr, len(addr))
	}

	decoded := make([]byte, 25)
	for _, c := range addr {
		val := strings.IndexRune(base58Alphabet, c)
		if val < 0 {
			return "", fmt.Errorf("address %q contains non-base58 character %q", addr, c)
		}
		carry := val
		for i := 24; i >= 0; i-- {
			carry += 58 * int(decoded[i])
			decoded[i] = byte(carry % 256)
			carry /= 256
		}
		if carry != 0 {
			return "", fmt.Errorf("address %q does not fit in a 25-byte payload", addr)
		}
	}

	// This puzzle set only uses P2PKH (version byte 0x00, leading '1').
	if decoded[0] != 0x00 {
		return "", fmt.Errorf("address %q is not P2PKH (version byte 0x%02x)", addr, decoded[0])
	}

	h160 := make([]byte, 0, 40)
	for i := 1; i <= 20; i++ {
		h160 = append(h160, fmt.Sprintf("%02x", decoded[i])...)
	}
	return string(h160), nil
}

// ========== Main ==========

func main() {
	puzzleFile := "../puzzles.json"
	if p := os.Getenv("PUZZLES_FILE"); p != "" {
		puzzleFile = p
	}
	data, err := os.ReadFile(puzzleFile)
	if err != nil {
		log.Fatalf("Cannot read puzzles.json: %v", err)
	}
	var puzzles []PuzzleEntry
	json.Unmarshal(data, &puzzles)
	log.Printf("Loaded %d puzzles", len(puzzles))

	puzzleNum := 71
	if p := os.Getenv("PUZZLE_NUM"); p != "" {
		fmt.Sscanf(p, "%d", &puzzleNum)
	}

	// chunk_bits = min(puzzle_range - 1, 30)
	chunkBits := puzzleNum - 1
	if chunkBits > 30 {
		chunkBits = 30
	}

	port := "8080"
	if p := os.Getenv("PORT"); p != "" {
		port = p
	}
	dbPath := "puzzle_pool.db"
	if p := os.Getenv("DB_PATH"); p != "" {
		dbPath = p
	}

	// Seed math/rand (not needed, we use crypto/rand for allocation)

	coord, err := NewCoordinator(dbPath, puzzles, puzzleNum, chunkBits)
	if err != nil {
		log.Fatalf("Failed: %v", err)
	}

	http.HandleFunc("/api/register", coord.handleRegister)
	http.HandleFunc("/api/tasks", coord.handleGetTasks)
	http.HandleFunc("/api/submit", coord.handleSubmit)
	http.HandleFunc("/api/stats", coord.handleStats)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })

	puzzle := puzzles[puzzleNum-1]
	log.Printf("=== Bitcoin Puzzle Pool Coordinator ===")
	log.Printf("Target:  Puzzle #%d (%s)", puzzleNum, puzzle.Addr)
	log.Printf("Chunk:   2^%d = %d keys/task", chunkBits, coord.chunkSize)
	log.Printf("Total:   %s chunks", coord.totalChunks.String())
	log.Printf("H160:    %s", coord.targetH160)
	log.Printf("Listen:  :%s", port)

	// Start puzzle monitor: immediate found-key alerts, the daily report, and
	// the solved-detection + auto-switch loop.
	emailTo := os.Getenv("EMAIL_TO")
	if emailTo == "" {
		emailTo = "359207423@qq.com"
	}

	solveCheck := 10 * time.Minute
	if v := os.Getenv("SOLVE_CHECK_MINUTES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			solveCheck = time.Duration(n) * time.Minute
		} else {
			log.Printf("Ignoring invalid SOLVE_CHECK_MINUTES=%q", v)
		}
	}

	reportAt := 9 * time.Hour
	if v := os.Getenv("REPORT_AT_UTC"); v != "" {
		d, err := parseClock(v)
		if err != nil {
			log.Fatalf("Invalid REPORT_AT_UTC: %v", err)
		}
		reportAt = d
	}

	log.Printf("Alerts: to %s | daily report %s UTC | solve check every %s",
		emailTo, formatClock(reportAt), solveCheck)

	monitor := NewPuzzleMonitor(coord, emailTo, solveCheck, reportAt)
	monitor.Start()

	log.Fatal(http.ListenAndServe(":"+port, nil))
}
