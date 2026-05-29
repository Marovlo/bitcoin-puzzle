#pragma once
#include <cstdint>
#include <array>
#include <string>
#include <memory>
#include <atomic>

// Compute backend interface — all backends implement this
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual bool init() = 0;
    virtual std::string name() const = 0;

    // Search [start_lo:start_hi, start+size) for target_h160.
    // Returns true if found, sets found_lo/found_hi.
    virtual bool search(uint64_t start_lo, uint64_t start_hi, uint64_t size,
                        const uint8_t target_h160[20],
                        uint64_t& found_lo, uint64_t& found_hi) = 0;

    // Keys/second estimate
    virtual uint64_t benchmark(uint64_t sample_size) = 0;

    // Set external stop flag (checked periodically during search)
    virtual void set_stop_flag(std::atomic<bool>*) {}
};

// Task from coordinator
struct PoolTask {
    int64_t id;
    uint64_t chunk_index;
    uint64_t start_hi;
    uint64_t start_lo;
    uint64_t size;
    std::string target_h160_hex;
};

// Worker configuration
struct WorkerConfig {
    std::string coordinator_url;  // e.g. http://your-server:8080
    std::string worker_id;
    std::string backend_name;     // metal, cpu, cpu_avx512, cuda
    std::string hostname;
};
