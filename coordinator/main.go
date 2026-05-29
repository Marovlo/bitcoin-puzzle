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
	WorkerID string           `json:"worker_id"`
	Results  []SubmitResult   `json:"results"`
}

type SubmitResult struct {
	TaskID int64  `json:"task_id"`
	Found  bool   `json:"found"`
	KeyHex string `json:"key_hex,omitempty"`
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

	// Concurrency: mutex protects the in-memory allocation state
	mu       sync.Mutex
	workers  map[string]time.Time // worker_id -> last_seen

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
	// Connection pool settings for concurrency
	db.SetMaxOpenConns(1) // SQLite only supports 1 writer anyway
	db.SetMaxIdleConns(1)

	c := &Coordinator{
		db:        db,
		puzzles:   puzzles,
		puzzleNum: puzzleNum,
		chunkBits: chunkBits,
		workers:   make(map[string]time.Time),
		startTime: time.Now(),
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
	c.targetH160 = addressToH160Hex(puzzle.Addr)

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
	return err
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
// Holds the mutex for the entire operation to guarantee no duplicates.
func (c *Coordinator) allocateChunks(workerID string, count int) ([]Task, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now().Unix()
	c.workers[workerID] = time.Now()

	totalU64 := c.totalChunks.Uint64() // safe for puzzles up to ~93 bits of chunks
	// For puzzles where totalChunks > 2^63, we use big.Int random
	useBigRand := !c.totalChunks.IsUint64()

	var tasks []Task
	maxAttempts := count * 50 // up to 50 retries per requested task

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

		// Compute start key
		offset := new(big.Int).SetUint64(idx)
		offset.Mul(offset, new(big.Int).SetUint64(c.chunkSize))
		startKey := new(big.Int).Add(c.rangeStart, offset)
		startHex := fmt.Sprintf("%x", startKey)

		res, err := c.stmtInsertTask.Exec(idx, startHex, c.chunkSize, workerID, now)
		if err != nil {
			continue // race condition, another goroutine got it
		}
		rows, _ := res.RowsAffected()
		if rows == 0 {
			continue // UNIQUE violation, already assigned
		}
		id, _ := res.LastInsertId()
		tasks = append(tasks, Task{
			ID: id, ChunkIndex: idx, StartHex: startHex,
			Size: c.chunkSize, Status: 1, WorkerID: workerID, AssignedAt: now,
		})
	}

	if len(tasks) == 0 {
		// Try to reclaim stale tasks
		return c.reclaimStale(workerID, count)
	}
	return tasks, nil
}

func (c *Coordinator) reclaimStale(workerID string, count int) ([]Task, error) {
	now := time.Now().Unix()
	rows, err := c.db.Query(
		`SELECT id, chunk_index, start_hex, size FROM active_tasks WHERE worker_id='__stale__' LIMIT ?`,
		count)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tasks []Task
	for rows.Next() {
		var t Task
		rows.Scan(&t.ID, &t.ChunkIndex, &t.StartHex, &t.Size)
		c.db.Exec(`UPDATE active_tasks SET worker_id=?, assigned_at=? WHERE id=?`,
			workerID, now, t.ID)
		t.WorkerID = workerID
		t.AssignedAt = now
		t.Status = 1
		tasks = append(tasks, t)
	}
	if len(tasks) == 0 {
		return nil, fmt.Errorf("no tasks available")
	}
	return tasks, nil
}

func (c *Coordinator) completeTasks(workerID string, results []SubmitResult) (int, string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.workers[workerID] = time.Now()
	completed := 0
	foundKey := ""

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
			foundKey = r.KeyHex
			c.db.Exec(`INSERT OR REPLACE INTO found_keys (puzzle_num, key_hex, worker_id) VALUES (?, ?, ?)`,
				c.puzzleNum, r.KeyHex, workerID)
			log.Printf("!!! KEY FOUND: puzzle #%d key=%s worker=%s", c.puzzleNum, r.KeyHex, workerID)
		}
	}
	return completed, foundKey
}

func (c *Coordinator) reapLoop() {
	for range time.NewTicker(60 * time.Second).C {
		cutoff := time.Now().Add(-10 * time.Minute).Unix()
		c.mu.Lock()
		c.db.Exec(`UPDATE active_tasks SET worker_id='__stale__', assigned_at=0 WHERE assigned_at > 0 AND assigned_at < ?`, cutoff)
		c.mu.Unlock()
	}
}

// Switch to a new puzzle target. Called by monitor when current puzzle is solved.
// Workers will get the new puzzle info on their next task fetch.
func (c *Coordinator) switchPuzzle(newPuzzleNum int) {
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
	c.targetH160 = addressToH160Hex(c.puzzles[newPuzzleNum-1].Addr)
	c.completedCount = 0

	// Reset done table for new puzzle (old data is no longer relevant)
	c.db.Exec(`DELETE FROM done`)

	log.Printf("[Switch] Now targeting puzzle #%d (%s)", newPuzzleNum, c.puzzles[newPuzzleNum-1].Addr)
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
	completed, foundKey := c.completeTasks(req.WorkerID, req.Results)
	resp := map[string]interface{}{"status": "ok", "completed": completed}
	if foundKey != "" {
		resp["found_key"] = foundKey
	}
	json.NewEncoder(w).Encode(resp)
}

func (c *Coordinator) handleStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(c.getStats())
}

// ========== Helpers ==========

func addressToH160Hex(addr string) string {
	b58 := "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
	decoded := make([]byte, 25)
	for _, c := range addr {
		val := 0
		for i, b := range b58 {
			if b == c {
				val = i
				break
			}
		}
		carry := val
		for i := 24; i >= 0; i-- {
			carry += 58 * int(decoded[i])
			decoded[i] = byte(carry % 256)
			carry /= 256
		}
	}
	h160 := ""
	for i := 1; i <= 20; i++ {
		h160 += fmt.Sprintf("%02x", decoded[i])
	}
	return h160
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

	// Start puzzle monitor (auto-switch + daily email)
	emailTo := os.Getenv("EMAIL_TO")
	if emailTo == "" {
		emailTo = "359207423@qq.com"
	}
	monitor := NewPuzzleMonitor(coord, emailTo)
	monitor.Start()

	log.Fatal(http.ListenAndServe(":"+port, nil))
}
