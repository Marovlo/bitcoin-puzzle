#pragma once

#ifdef __APPLE__

#include "worker.h"
#include "kernels/metal_solver.h"
#include <chrono>
#include <cstring>

class MetalBackend : public ComputeBackend {
public:
    static inline std::atomic<bool>* g_stop_flag = nullptr;
    void set_stop_flag(std::atomic<bool>* flag) { g_stop_flag = flag; }

    bool init() override {
        return solver_.init();
    }

    std::string name() const override {
        return "metal_" + solver_.device_name();
    }

    void set_batch(uint64_t batch_size) {
        solver_.set_batch_size(batch_size);
    }

    bool search(uint64_t start_lo, uint64_t start_hi, uint64_t size,
                const uint8_t target_h160[20],
                uint64_t& found_lo, uint64_t& found_hi) override {
        std::array<uint8_t, 20> target;
        memcpy(target.data(), target_h160, 20);
        solver_.set_target(target);

        uint64_t batch = solver_.batch_size();
        uint64_t remaining = size;
        uint64_t off_lo = start_lo;
        uint64_t off_hi = start_hi;

        while (remaining > 0) {
            if (g_stop_flag && !g_stop_flag->load(std::memory_order_relaxed)) return false;
            uint64_t bs = std::min(remaining, batch);
            auto result = solver_.search_batch(off_lo, off_hi, bs);
            if (result.found) {
                found_lo = result.private_key.d[0];
                found_hi = result.private_key.d[1];
                return true;
            }
            // Advance offset
            uint64_t new_lo = off_lo + bs;
            if (new_lo < off_lo) off_hi++;  // carry
            off_lo = new_lo;
            remaining -= bs;
        }
        return false;
    }

    uint64_t benchmark(uint64_t sample_size) override {
        // Metal GPU needs large batches to saturate. Use at least 4M keys
        // (one full dispatch) for accurate measurement.
        uint64_t actual = std::max(sample_size, (uint64_t)4'000'000);
        auto result = solver_.benchmark(actual);
        return (uint64_t)result.keys_per_second;
    }

private:
    MetalSolver solver_;
};

#endif  // __APPLE__
