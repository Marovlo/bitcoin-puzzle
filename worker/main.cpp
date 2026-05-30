// Bitcoin Puzzle Pool Worker - Pipeline Architecture
//
// Three async stages, never blocking compute:
//   [Fetch Thread] → task_queue → [Compute Thread] → result_queue → [Submit Thread]
//
// - Fetch thread: pre-fetches batch tasks, keeps task_queue full
// - Compute thread: takes tasks from queue, runs GPU/CPU, pushes results
// - Submit thread: batches and sends completed results to coordinator
//
// Compute never waits on network. Only blocked if task_queue is empty
// (fetch can't keep up, which means network is the bottleneck anyway).

#include "worker.h"
#include "http_client.h"
#include "json_helpers.h"
#include "backend_cpu.h"
#include "backend_multi.h"
#ifdef __APPLE__
#include "backend_metal.h"
#endif
#ifdef USE_CUDA
#include "backend_cuda.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "platform.h"

// ========== Thread-safe queue ==========

template<typename T>
class TSQueue {
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mu_);
        q_.push(std::move(item));
        cv_.notify_one();
    }

    // Blocking pop with timeout. Returns false if timed out.
    bool pop(T& item, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mu_);
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                          [&]{ return !q_.empty(); }))
            return false;
        item = std::move(q_.front());
        q_.pop();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return q_.size();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::queue<T> q_;
};

// ========== Task / Result types ==========

struct ComputeTask {
    int64_t id;
    uint64_t chunk_index;
    uint64_t start_lo;
    uint64_t start_hi;
    uint64_t size;
};

struct ComputeResult {
    int64_t task_id;
    bool found;
    std::string key_hex;
};

// ========== Globals ==========

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running = false; }

static void hex_to_bytes(const std::string& hex, uint8_t* out, size_t len) {
    memset(out, 0, len);
    size_t hex_len = hex.size();
    for (size_t i = 0; i + 1 < hex_len; i += 2) {
        unsigned int b;
        sscanf(hex.c_str() + i, "%2x", &b);
        out[i / 2] = (uint8_t)b;
    }
}

static void parse_hex_key(const std::string& hex, uint64_t& hi, uint64_t& lo) {
    size_t len = hex.size();
    if (len <= 16) {
        lo = strtoull(hex.c_str(), nullptr, 16);
        hi = 0;
    } else {
        std::string hi_str = hex.substr(0, len - 16);
        std::string lo_str = hex.substr(len - 16);
        hi = strtoull(hi_str.c_str(), nullptr, 16);
        lo = strtoull(lo_str.c_str(), nullptr, 16);
    }
}

// ========== Fetch Thread ==========

static void fetch_thread_fn(const std::string& base_url, const std::string& worker_id,
                            int batch_count, TSQueue<ComputeTask>& task_queue,
                            uint8_t target_h160[20], std::atomic<bool>& target_ready) {
    const int MAX_QUEUE_SIZE = batch_count * 3; // Keep 3 batches buffered

    while (g_running) {
        // Don't over-fill the queue
        if ((int)task_queue.size() >= MAX_QUEUE_SIZE) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        int fetch_count = MAX_QUEUE_SIZE - (int)task_queue.size();
        if (fetch_count < 1) fetch_count = 1;
        if (fetch_count > batch_count) fetch_count = batch_count;

        char url[512];
        snprintf(url, sizeof(url), "%s/api/tasks?worker_id=%s&count=%d",
                 base_url.c_str(), worker_id.c_str(), fetch_count);

        auto resp = http::get(url);
        if (!resp.ok()) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }

        std::string error = json::get_string(resp.body, "error");
        if (!error.empty()) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }

        // Parse target_h160 — update on every response to detect puzzle switch
        {
            std::string h160_hex = json::get_string(resp.body, "target_h160");
            if (!h160_hex.empty()) {
                uint8_t new_target[20];
                hex_to_bytes(h160_hex, new_target, 20);
                if (!target_ready.load() || memcmp(new_target, target_h160, 20) != 0) {
                    memcpy(target_h160, new_target, 20);
                    target_ready.store(true);
                    int pnum = (int)json::get_int(resp.body, "puzzle_num");
                    if (pnum > 0) {
                        printf("  [fetch] Target updated: puzzle #%d h160=%s\n",
                               pnum, h160_hex.c_str());
                    }
                }
            }
        }

        // Parse tasks array
        auto tasks_pos = resp.body.find("\"tasks\"");
        if (tasks_pos == std::string::npos) continue;
        auto arr_start = resp.body.find('[', tasks_pos);
        auto arr_end = resp.body.find(']', arr_start);
        if (arr_start == std::string::npos || arr_end == std::string::npos) continue;
        std::string tasks_arr = resp.body.substr(arr_start, arr_end - arr_start + 1);

        size_t search_pos = 0;
        while (g_running) {
            auto obj_start = tasks_arr.find('{', search_pos);
            if (obj_start == std::string::npos) break;
            int depth = 0;
            size_t obj_end = obj_start;
            for (size_t i = obj_start; i < tasks_arr.size(); i++) {
                if (tasks_arr[i] == '{') depth++;
                else if (tasks_arr[i] == '}') { depth--; if (depth == 0) { obj_end = i; break; } }
            }
            std::string tj = tasks_arr.substr(obj_start, obj_end - obj_start + 1);
            search_pos = obj_end + 1;

            ComputeTask ct;
            ct.id = json::get_int(tj, "id");
            ct.chunk_index = json::get_uint(tj, "chunk_index");
            ct.size = json::get_uint(tj, "size");

            std::string start_hex = json::get_string(tj, "start_hex");
            parse_hex_key(start_hex, ct.start_hi, ct.start_lo);

            task_queue.push(ct);
        }
    }
}

// ========== Submit Thread ==========

static void submit_thread_fn(const std::string& base_url, const std::string& worker_id,
                             TSQueue<ComputeResult>& result_queue,
                             std::atomic<uint64_t>& tasks_submitted) {
    std::vector<ComputeResult> batch;

    while (true) {
        ComputeResult r;
        bool got = result_queue.pop(r, 500);

        if (!got) {
            // Timeout: flush accumulated batch
            if (!batch.empty()) goto flush;
            // If shutting down and queue is empty, we're done
            if (!g_running && result_queue.size() == 0) break;
            continue;
        }
        batch.push_back(r);

        // Flush when batch full or key found
        if (batch.size() >= 10 || r.found) {
            goto flush;
        }
        continue;

    flush:
        {
            std::string json_str = "{\"worker_id\":\"" + worker_id + "\",\"results\":[";
            for (size_t i = 0; i < batch.size(); i++) {
                if (i > 0) json_str += ",";
                char buf[256];
                snprintf(buf, sizeof(buf), "{\"task_id\":%lld,\"found\":%s,\"key_hex\":\"%s\"}",
                         (long long)batch[i].task_id,
                         batch[i].found ? "true" : "false",
                         batch[i].key_hex.c_str());
                json_str += buf;
            }
            json_str += "]}";
            auto resp = http::post(base_url + "/api/submit", json_str);
            if (resp.ok()) {
                tasks_submitted.fetch_add(batch.size());
                printf("  [submit] %zu tasks uploaded OK\n", batch.size());
            } else {
                printf("  [submit] FAILED (status=%d), will retry\n", resp.status_code);
                // On failure, DON'T clear batch — retry on next loop
                continue;
            }
            batch.clear();
        }

        // After flush, check if we should exit
        if (!g_running && result_queue.size() == 0) break;
    }

    // Final flush of any remaining (with retry)
    if (!batch.empty()) {
        for (int retry = 0; retry < 3; retry++) {
            std::string json_str = "{\"worker_id\":\"" + worker_id + "\",\"results\":[";
            for (size_t i = 0; i < batch.size(); i++) {
                if (i > 0) json_str += ",";
                char buf[256];
                snprintf(buf, sizeof(buf), "{\"task_id\":%lld,\"found\":%s,\"key_hex\":\"%s\"}",
                         (long long)batch[i].task_id,
                         batch[i].found ? "true" : "false",
                         batch[i].key_hex.c_str());
                json_str += buf;
            }
            json_str += "]}";
            auto resp = http::post(base_url + "/api/submit", json_str);
            if (resp.ok()) {
                tasks_submitted.fetch_add(batch.size());
                printf("  [submit] Final flush: %zu tasks uploaded OK\n", batch.size());
                break;
            }
            printf("  [submit] Final flush retry %d/3 failed\n", retry + 1);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

// ========== Main ==========

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0); // Line-buffered output (for nohup/redirect)
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Config
    std::string coordinator_url = "http://localhost:8080";
    std::string backend_name = "auto";
    char id_buf[32];
    srand((unsigned)time(nullptr) ^ (unsigned)getpid());
    snprintf(id_buf, sizeof(id_buf), "w_%08x", (unsigned)rand());
    std::string worker_id = id_buf;
    char hname[256] = "unknown";
    gethostname_compat(hname, sizeof(hname));

    bool test_mode = false;
    int cpu_threads = 0;        // 0 = auto
    uint64_t metal_batch = 0;   // 0 = default (4M)

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--url") == 0 || strcmp(argv[i], "-u") == 0) && i + 1 < argc)
            coordinator_url = argv[++i];
        else if ((strcmp(argv[i], "--backend") == 0 || strcmp(argv[i], "-b") == 0) && i + 1 < argc)
            backend_name = argv[++i];
        else if (strcmp(argv[i], "--id") == 0 && i + 1 < argc)
            worker_id = argv[++i];
        else if (strcmp(argv[i], "--cpu-threads") == 0 && i + 1 < argc)
            cpu_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--metal-batch") == 0 && i + 1 < argc)
            metal_batch = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "--test") == 0 || strcmp(argv[i], "-t") == 0)
            test_mode = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  -u, --url URL          Coordinator URL (default: http://localhost:8080)\n");
            printf("  -b, --backend NAME     Backend: auto, metal, cpu, cuda (default: auto)\n");
            printf("      --id ID            Worker ID (default: random)\n");
            printf("      --cpu-threads N    CPU thread count (default: all cores)\n");
            printf("      --metal-batch N    Metal keys per dispatch (default: 4000000)\n");
            printf("  -t, --test             Run self-test and exit\n");
            printf("  -h, --help             Show help\n");
            return 0;
        }
    }

    // Init backend
    std::unique_ptr<ComputeBackend> backend;
    if (backend_name == "auto" || backend_name == "multi") {
        backend = std::make_unique<MultiBackend>(cpu_threads, metal_batch);
    } else if (backend_name == "metal") {
#ifdef __APPLE__
        auto m = std::make_unique<MetalBackend>();
        if (metal_batch > 0) m->set_batch(metal_batch);
        backend = std::move(m);
#else
        printf("[!] Metal not available\n"); return 1;
#endif
    } else if (backend_name == "cpu") {
        backend = std::make_unique<CPUBackend>(cpu_threads);
#ifdef USE_CUDA
    } else if (backend_name == "cuda") {
        backend = std::make_unique<CUDABackend>(0, metal_batch);
#endif
#ifdef USE_HIP
    } else if (backend_name == "hip") {
        backend = std::make_unique<HIPBackend>(0);
#endif
    } else {
        printf("[!] Unknown backend: %s\n", backend_name.c_str());
        printf("    Available: auto, cpu");
#ifdef __APPLE__
        printf(", metal");
#endif
#ifdef USE_CUDA
        printf(", cuda");
#endif
#ifdef USE_HIP
        printf(", hip");
#endif
        printf("\n");
        return 1;
    }

    if (!backend->init()) { printf("[!] Backend init failed\n"); return 1; }
    backend->set_stop_flag(&g_running); // Let backends check for Ctrl+C

    printf("=== Bitcoin Puzzle Pool Worker%s ===\n", test_mode ? " (TEST MODE)" : " (Pipeline)");
    printf("  Worker:    %s\n", worker_id.c_str());
    printf("  Backend:   %s\n", backend->name().c_str());
    printf("  Server:    %s\n", coordinator_url.c_str());

    // Benchmark
    uint64_t rate = backend->benchmark(50000);
    printf("  Speed:     %.2f MKeys/s\n", rate / 1e6);

    // ========== TEST MODE ==========
    if (test_mode) {
        printf("\n--- Self-Test ---\n");
        int pass = 0, fail = 0;

        // Test 1: Health check
        printf("[1/5] Health check... ");
        {
            auto resp = http::get(coordinator_url + "/health");
            if (resp.ok()) { printf("OK\n"); pass++; }
            else { printf("FAIL (status=%d)\n", resp.status_code); fail++; }
        }

        // Test 2: Register
        printf("[2/5] Register... ");
        {
            char buf[512];
            snprintf(buf, sizeof(buf), R"({"worker_id":"%s","backend":"%s","hostname":"%s","rate":%llu})",
                     worker_id.c_str(), backend->name().c_str(), hname, (unsigned long long)rate);
            auto resp = http::post(coordinator_url + "/api/register", buf);
            if (resp.ok()) { printf("OK\n"); pass++; }
            else { printf("FAIL (status=%d body=%s)\n", resp.status_code, resp.body.c_str()); fail++; }
        }

        // Test 3: Get task
        printf("[3/5] Get task... ");
        int64_t task_id = 0;
        uint64_t task_start_lo = 0, task_start_hi = 0, task_size = 0;
        std::string target_h160_hex;
        {
            char url[512];
            snprintf(url, sizeof(url), "%s/api/tasks?worker_id=%s&count=1", coordinator_url.c_str(), worker_id.c_str());
            auto resp = http::get(url);
            if (!resp.ok()) {
                printf("FAIL (status=%d)\n", resp.status_code); fail++;
            } else {
                std::string err = json::get_string(resp.body, "error");
                if (!err.empty()) {
                    printf("FAIL (error=%s)\n", err.c_str()); fail++;
                } else {
                    target_h160_hex = json::get_string(resp.body, "target_h160");
                    auto tasks_pos = resp.body.find("\"tasks\"");
                    auto arr_start = resp.body.find('[', tasks_pos);
                    auto obj_start = resp.body.find('{', arr_start);
                    int depth = 0; size_t obj_end = obj_start;
                    for (size_t x = obj_start; x < resp.body.size(); x++) {
                        if (resp.body[x] == '{') depth++;
                        else if (resp.body[x] == '}') { depth--; if (depth==0) { obj_end=x; break; } }
                    }
                    std::string tj = resp.body.substr(obj_start, obj_end - obj_start + 1);
                    task_id = json::get_int(tj, "id");
                    task_size = json::get_uint(tj, "size");
                    std::string sh = json::get_string(tj, "start_hex");
                    parse_hex_key(sh, task_start_hi, task_start_lo);
                    printf("OK (task_id=%lld, size=%llu, target=%s)\n",
                           (long long)task_id, (unsigned long long)task_size,
                           target_h160_hex.substr(0, 8).c_str());
                    pass++;
                }
            }
        }

        // Test 4: Compute (small sample)
        printf("[4/5] Compute (10K keys)... ");
        if (task_size > 0 && !target_h160_hex.empty()) {
            uint8_t target[20];
            hex_to_bytes(target_h160_hex, target, 20);
            uint64_t fl = 0, fh = 0;
            auto t0 = std::chrono::steady_clock::now();
            // Only compute 10K keys (not the full chunk) for speed
            backend->search(task_start_lo, task_start_hi, 10000, target, fl, fh);
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            printf("OK (%.3fs, %.1f MK/s)\n", elapsed, 10000.0 / elapsed / 1e6);
            pass++;
        } else {
            printf("SKIP (no task)\n");
        }

        // Test 5: Submit result
        printf("[5/5] Submit... ");
        if (task_id > 0) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     R"({"worker_id":"%s","results":[{"task_id":%lld,"found":false,"key_hex":""}]})",
                     worker_id.c_str(), (long long)task_id);
            auto resp = http::post(coordinator_url + "/api/submit", buf);
            if (resp.ok() && resp.body.find("\"ok\"") != std::string::npos) {
                printf("OK (response: %s)\n", resp.body.c_str());
                pass++;
            } else {
                printf("FAIL (status=%d body=%s)\n", resp.status_code, resp.body.c_str());
                fail++;
            }
        } else {
            printf("SKIP (no task)\n");
        }

        // Verify coordinator recorded it
        printf("\n--- Verify ---\n");
        {
            auto resp = http::get(coordinator_url + "/api/stats");
            if (resp.ok()) {
                printf("  Stats: %s\n", resp.body.c_str());
            }
        }

        printf("\n--- Result: %d passed, %d failed ---\n", pass, fail);
        return fail > 0 ? 1 : 0;
    }

    // ========== NORMAL MODE (Pipeline) ==========

    // Register
    {
        char buf[512];
        snprintf(buf, sizeof(buf), R"({"worker_id":"%s","backend":"%s","hostname":"%s","rate":%llu})",
                 worker_id.c_str(), backend->name().c_str(), hname, (unsigned long long)rate);
        auto resp = http::post(coordinator_url + "/api/register", buf);
        if (!resp.ok()) {
            printf("[!] Registration failed. Is coordinator at %s?\n", coordinator_url.c_str());
            return 1;
        }
    }
    printf("  Registered OK\n\n");

    // Auto-determine batch_count: aim for ~5 min of work per fetch
    int batch_count = 8; // will auto-adjust

    // Shared state
    TSQueue<ComputeTask> task_queue;
    TSQueue<ComputeResult> result_queue;
    uint8_t target_h160[20] = {};
    std::atomic<bool> target_ready{false};
    std::atomic<uint64_t> tasks_submitted{0};
    std::atomic<uint64_t> total_keys{0};

    // Launch fetch thread
    std::thread fetcher(fetch_thread_fn, coordinator_url, worker_id,
                        batch_count, std::ref(task_queue),
                        target_h160, std::ref(target_ready));

    // Launch submit thread
    std::thread submitter(submit_thread_fn, coordinator_url, worker_id,
                          std::ref(result_queue), std::ref(tasks_submitted));

    // Compute loop (main thread = compute thread for simplicity)
    printf("[*] Pipeline started: fetch | compute | submit\n");
    printf("[*] Waiting for first task...\n");

    // Wait for target h160 to be ready
    while (!target_ready.load() && g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    uint64_t tasks_computed = 0;
    auto t_start = std::chrono::steady_clock::now();

    while (g_running) {
        ComputeTask ct;
        if (!task_queue.pop(ct, 1000)) {
            continue; // queue empty, wait for fetch
        }

        auto t0 = std::chrono::steady_clock::now();
        uint64_t found_lo = 0, found_hi = 0;
        bool found = backend->search(ct.start_lo, ct.start_hi, ct.size,
                                     target_h160, found_lo, found_hi);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double mks = (elapsed > 0) ? ct.size / elapsed / 1e6 : 0;

        tasks_computed++;
        total_keys.fetch_add(ct.size);

        // Format result
        ComputeResult cr;
        cr.task_id = ct.id;
        cr.found = found;
        if (found) {
            char kbuf[64];
            if (found_hi > 0) snprintf(kbuf, sizeof(kbuf), "%llx%016llx",
                                       (unsigned long long)found_hi, (unsigned long long)found_lo);
            else snprintf(kbuf, sizeof(kbuf), "%llx", (unsigned long long)found_lo);
            cr.key_hex = kbuf;
        }
        result_queue.push(cr);

        printf("  [%llu] chunk=%llu %.1fs %.1f MK/s q=%zu %s\n",
               (unsigned long long)tasks_computed,
               (unsigned long long)ct.chunk_index,
               elapsed, mks, task_queue.size(),
               found ? "*** FOUND ***" : "");

        if (found) {
            printf("\n!!! KEY FOUND: %s !!!\n", cr.key_hex.c_str());
            g_running = false;
            break;
        }

        // Auto-calibrate batch_count after first task
        if (tasks_computed == 1 && elapsed > 0) {
            int ideal = (int)(300.0 / elapsed);
            if (ideal < 2) ideal = 2;
            if (ideal > 50) ideal = 50;
            batch_count = ideal;
            printf("  [auto] batch_count=%d (%.0fs/task)\n", batch_count, elapsed);
        }
    }

    // Graceful shutdown: ensure all completed results are uploaded
    g_running = false;
    printf("\n[*] Shutting down... flushing pending results\n");

    // Give submit thread time to drain result_queue
    for (int i = 0; i < 20 && result_queue.size() > 0; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Wait for fetch thread to stop
    fetcher.join();
    // Wait for submit thread to finish flushing
    submitter.join();

    printf("[+] All completed tasks submitted to coordinator\n");

    auto t_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("\n=== Summary ===\n");
    printf("  Computed: %llu tasks\n", (unsigned long long)tasks_computed);
    printf("  Submitted: %llu tasks\n", (unsigned long long)tasks_submitted.load());
    printf("  Keys: %llu\n", (unsigned long long)total_keys.load());
    printf("  Time: %.1fs\n", total_time);
    if (total_time > 0)
        printf("  Rate: %.2f MKeys/s\n", total_keys.load() / total_time / 1e6);
    return 0;
}
