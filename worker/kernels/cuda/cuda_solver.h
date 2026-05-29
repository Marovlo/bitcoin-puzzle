#pragma once
#ifdef USE_CUDA

#include <cstdint>
#include <array>
#include <string>

class CUDASolver {
public:
    CUDASolver();
    ~CUDASolver();

    bool init(int device_id = 0);
    std::string device_name() const;

    bool set_target(const std::array<uint8_t, 20>& hash160);
    void set_batch_size(uint64_t bs);
    uint64_t batch_size() const;

    // Search batch, returns true if key found
    bool search_batch(uint64_t start_lo, uint64_t start_hi, uint64_t batch_size,
                      uint64_t& found_lo, uint64_t& found_hi);

    // Benchmark
    double benchmark(uint64_t num_keys); // returns keys/sec

private:
    struct Impl;
    Impl* impl_;
};

#endif // USE_CUDA
