#pragma once

#ifdef USE_CUDA

#include "worker.h"
#include "kernels/cuda/cuda_solver.h"
#include <chrono>
#include <cstring>
#include <algorithm>

class CUDABackend : public ComputeBackend {
public:
    CUDABackend(int device_id = 0, uint64_t batch = 0)
        : device_id_(device_id), batch_(batch) {}

    bool init() override {
        if (!solver_.init(device_id_)) return false;
        if (batch_ > 0) solver_.set_batch_size(batch_);
        return true;
    }

    std::string name() const override {
        return "cuda_" + solver_.device_name();
    }

    void set_batch(uint64_t bs) { solver_.set_batch_size(bs); }

    bool search(uint64_t start_lo, uint64_t start_hi, uint64_t size,
                const uint8_t target_h160[20],
                uint64_t& found_lo, uint64_t& found_hi) override {
        std::array<uint8_t, 20> target;
        memcpy(target.data(), target_h160, 20);
        solver_.set_target(target);

        uint64_t batch = solver_.batch_size();
        uint64_t remaining = size;
        uint64_t off_lo = start_lo, off_hi = start_hi;

        while (remaining > 0) {
            uint64_t bs = std::min(remaining, batch);
            if (solver_.search_batch(off_lo, off_hi, bs, found_lo, found_hi))
                return true;
            uint64_t new_lo = off_lo + bs;
            if (new_lo < off_lo) off_hi++;
            off_lo = new_lo;
            remaining -= bs;
        }
        return false;
    }

    uint64_t benchmark(uint64_t sample_size) override {
        uint64_t actual = std::max(sample_size, solver_.batch_size());
        return (uint64_t)solver_.benchmark(actual);
    }

private:
    CUDASolver solver_;
    int device_id_;
    uint64_t batch_;
};

#endif // USE_CUDA
