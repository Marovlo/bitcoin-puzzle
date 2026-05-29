#pragma once

#ifdef USE_CUDA

#include "worker.h"
#include <cstdio>

// CUDA backend stub — implement when CUDA device is available.
// The kernel logic is identical to puzzle.metal, ported to .cu:
//   - Replace `ulong` with `uint64_t`
//   - Replace `mul128` with `__int128` or `__umul64hi`
//   - Replace Metal atomic with `atomicCAS`
//   - Use `<<<blocks, threads>>>` launch syntax
//
// Expected performance: RTX 4090 ~2-5 GKeys/s (200-500x over M3 Metal)

class CUDABackend : public ComputeBackend {
public:
    bool init() override {
        // TODO: cudaGetDevice, cudaMalloc, upload G table, compile kernel
        fprintf(stderr, "[CUDA] Backend not yet implemented. Build with -DUSE_CUDA and implement.\n");
        return false;
    }

    std::string name() const override { return "cuda"; }

    bool search(uint64_t start_lo, uint64_t start_hi, uint64_t size,
                const uint8_t target_h160[20],
                uint64_t& found_lo, uint64_t& found_hi) override {
        (void)start_lo; (void)start_hi; (void)size;
        (void)target_h160; (void)found_lo; (void)found_hi;
        return false;
    }

    uint64_t benchmark(uint64_t sample_size) override {
        (void)sample_size;
        return 0;
    }
};

#endif  // USE_CUDA
