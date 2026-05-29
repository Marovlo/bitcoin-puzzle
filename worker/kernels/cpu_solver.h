#pragma once
#include "types.h"

// CPU-based brute force solver
class CPUSolver {
public:
    CPUSolver();
    ~CPUSolver();

    bool init();

    // Search a batch of keys starting from (start_lo, start_hi)
    // Returns true if found, populates result
    SearchResult search_batch(uint64_t start_lo, uint64_t start_hi,
                              uint64_t batch_size,
                              const std::array<uint8_t, 20>& target_h160);

    // Run benchmark
    BenchmarkResult benchmark(uint64_t num_keys);

    // Verify a single key produces the expected hash160
    bool verify_key(uint64_t priv_lo, uint64_t priv_hi,
                    const std::array<uint8_t, 20>& expected_h160);

private:
    uint64_t* g_table_ = nullptr;
};
