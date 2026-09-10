#pragma once
#ifdef USE_OPENCL

#include "worker.h"
#include "kernels/opencl/opencl_solver.h"
#include <chrono>
#include <cstring>
#include <algorithm>

// ComputeBackend adapter over OpenCLSolver. Same shape as the HIP/CUDA backends.
class OpenCLBackend : public ComputeBackend {
public:
    explicit OpenCLBackend(int device_id = 0) : device_id_(device_id) {}

    bool init() override { return solver_.init(device_id_); }
    std::string name() const override { return "opencl_" + solver_.device_name(); }

    bool search(uint64_t start_lo, uint64_t start_hi, uint64_t size,
                const uint8_t target_h160[20],
                uint64_t& found_lo, uint64_t& found_hi) override {
        std::array<uint8_t, 20> target;
        memcpy(target.data(), target_h160, 20);
        solver_.set_target(target);

        const uint64_t batch = solver_.batch_size();
        uint64_t rem = size, off_lo = start_lo, off_hi = start_hi;
        while (rem > 0) {
            const uint64_t bs = std::min(rem, batch);
            if (solver_.search_batch(off_lo, off_hi, bs, found_lo, found_hi)) return true;
            const uint64_t new_lo = off_lo + bs;
            if (new_lo < off_lo) off_hi++;  // 128-bit carry
            off_lo = new_lo;
            rem -= bs;
        }
        return false;
    }

    uint64_t benchmark(uint64_t sample_size) override {
        return (uint64_t)solver_.benchmark(std::max(sample_size, solver_.batch_size()));
    }

private:
    OpenCLSolver solver_;
    int device_id_;
};

#endif // USE_OPENCL
