#ifdef USE_CUDA
#include "cuda_solver.h"
#include "../secp256k1.h"

#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <vector>

// Kernel declaration (defined in puzzle_kernel.cu)
extern "C" __global__ void puzzle_search(
    const uint64_t* g_table, const uint8_t* target_h160,
    uint64_t start_lo, uint64_t start_hi, uint64_t total_keys,
    uint64_t* match_lo, uint64_t* match_hi, uint32_t* match_found);

struct CUDASolver::Impl {
    int device_id = 0;
    std::string device_name_str;
    uint64_t batch = 4'000'000;

    // Device memory
    uint64_t* d_gtable = nullptr;
    uint8_t*  d_target = nullptr;
    uint64_t* d_match_lo = nullptr;
    uint64_t* d_match_hi = nullptr;
    uint32_t* d_match_found = nullptr;

    int blocks = 256;
    int threads_per_block = 256;
};

CUDASolver::CUDASolver() : impl_(new Impl()) {}

CUDASolver::~CUDASolver() {
    if (impl_) {
        cudaFree(impl_->d_gtable);
        cudaFree(impl_->d_target);
        cudaFree(impl_->d_match_lo);
        cudaFree(impl_->d_match_hi);
        cudaFree(impl_->d_match_found);
        delete impl_;
    }
}

bool CUDASolver::init(int device_id) {
    impl_->device_id = device_id;

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0 || device_id >= device_count) {
        fprintf(stderr, "[CUDA] No CUDA device found\n");
        return false;
    }
    cudaSetDevice(device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    impl_->device_name_str = prop.name;

    // Optimal launch config
    impl_->threads_per_block = 256;
    impl_->blocks = prop.multiProcessorCount * 4; // Good occupancy

    printf("[CUDA] Device: %s (SM %d.%d, %d SMs, %d MB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, (int)(prop.totalGlobalMem / 1048576));

    // Allocate device memory
    size_t gtable_bytes = secp256k1::G_TABLE_ULONGS * sizeof(uint64_t);
    cudaMalloc(&impl_->d_gtable, gtable_bytes);
    cudaMalloc(&impl_->d_target, 20);
    cudaMalloc(&impl_->d_match_lo, sizeof(uint64_t));
    cudaMalloc(&impl_->d_match_hi, sizeof(uint64_t));
    cudaMalloc(&impl_->d_match_found, sizeof(uint32_t));

    if (!impl_->d_gtable || !impl_->d_target) {
        fprintf(stderr, "[CUDA] Memory allocation failed\n");
        return false;
    }

    // Build G table on host and upload
    std::vector<uint64_t> gtable_host(secp256k1::G_TABLE_ULONGS);
    secp256k1::build_g_table(gtable_host.data());
    cudaMemcpy(impl_->d_gtable, gtable_host.data(), gtable_bytes, cudaMemcpyHostToDevice);

    printf("[CUDA] G table uploaded (%zu KB)\n", gtable_bytes / 1024);
    return true;
}

std::string CUDASolver::device_name() const { return impl_->device_name_str; }
uint64_t CUDASolver::batch_size() const { return impl_->batch; }
void CUDASolver::set_batch_size(uint64_t bs) { if (bs > 0) impl_->batch = bs; }

bool CUDASolver::set_target(const std::array<uint8_t, 20>& hash160) {
    cudaMemcpy(impl_->d_target, hash160.data(), 20, cudaMemcpyHostToDevice);
    return true;
}

bool CUDASolver::search_batch(uint64_t start_lo, uint64_t start_hi, uint64_t batch_size,
                              uint64_t& found_lo, uint64_t& found_hi) {
    if (batch_size == 0) batch_size = impl_->batch;

    // Reset match flags
    uint64_t zero64 = 0;
    uint32_t zero32 = 0;
    cudaMemcpy(impl_->d_match_lo, &zero64, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(impl_->d_match_hi, &zero64, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(impl_->d_match_found, &zero32, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = impl_->threads_per_block;
    int blocks = (int)((batch_size + threads - 1) / threads);

    puzzle_search<<<blocks, threads>>>(
        impl_->d_gtable, impl_->d_target,
        start_lo, start_hi, batch_size,
        impl_->d_match_lo, impl_->d_match_hi, impl_->d_match_found);

    cudaDeviceSynchronize();

    // Check result
    uint32_t found = 0;
    cudaMemcpy(&found, impl_->d_match_found, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (found) {
        cudaMemcpy(&found_lo, impl_->d_match_lo, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&found_hi, impl_->d_match_hi, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        return true;
    }
    return false;
}

double CUDASolver::benchmark(uint64_t num_keys) {
    // Dummy target
    std::array<uint8_t, 20> dummy;
    memset(dummy.data(), 0xFF, 20);
    set_target(dummy);

    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t remaining = num_keys;
    uint64_t offset = 1;
    while (remaining > 0) {
        uint64_t bs = std::min(remaining, impl_->batch);
        uint64_t fl, fh;
        search_batch(offset, 0, bs, fl, fh);
        offset += bs;
        remaining -= bs;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    return (double)num_keys / elapsed;
}

#endif // USE_CUDA
