#pragma once
#include "types.h"
#include <string>

// Metal GPU-accelerated brute force solver
class MetalSolver {
public:
    MetalSolver();
    ~MetalSolver();

    bool init();

    // Set target hash160 for comparison
    bool set_target(const std::array<uint8_t, 20>& hash160);

    // Search a batch of keys. Returns true if match found.
    SearchResult search_batch(uint64_t start_lo, uint64_t start_hi,
                              uint64_t batch_size);

    // Run benchmark
    BenchmarkResult benchmark(uint64_t num_keys);

    // Verify single key (for correctness testing)
    bool verify_key(uint64_t priv_lo, uint64_t priv_hi,
                    const std::array<uint8_t, 20>& expected_h160);

    std::string device_name() const;
    std::string error() const;

    // Batch size control
    uint64_t batch_size() const;
    void set_batch_size(uint64_t bs);

private:
    struct Impl;
    Impl* impl_;
};
