#ifdef USE_HIP
#include "hip_solver.h"
#include "../secp256k1.h"

#include <hip/hip_runtime.h>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <vector>

extern "C" __global__ void puzzle_search(
    const uint64_t* g_table, const uint8_t* target_h160,
    uint64_t start_lo, uint64_t start_hi, uint64_t total_keys,
    uint64_t* match_lo, uint64_t* match_hi, uint32_t* match_found);

struct HIPSolver::Impl {
    int device_id = 0;
    std::string device_name_str;
    uint64_t batch = 4'000'000;
    uint64_t* d_gtable = nullptr;
    uint8_t*  d_target = nullptr;
    uint64_t* d_match_lo = nullptr;
    uint64_t* d_match_hi = nullptr;
    uint32_t* d_match_found = nullptr;
    int blocks = 256;
    int threads_per_block = 256;
};

HIPSolver::HIPSolver() : impl_(new Impl()) {}
HIPSolver::~HIPSolver() {
    if (impl_) {
        hipFree(impl_->d_gtable); hipFree(impl_->d_target);
        hipFree(impl_->d_match_lo); hipFree(impl_->d_match_hi); hipFree(impl_->d_match_found);
        delete impl_;
    }
}

bool HIPSolver::init(int device_id) {
    impl_->device_id = device_id;
    int count = 0;
    hipGetDeviceCount(&count);
    if (count == 0 || device_id >= count) { fprintf(stderr, "[HIP] No device\n"); return false; }
    hipSetDevice(device_id);

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id);
    impl_->device_name_str = prop.name;
    impl_->threads_per_block = 256;
    impl_->blocks = prop.multiProcessorCount * 4;
    impl_->batch = (uint64_t)impl_->blocks * impl_->threads_per_block * 4;

    printf("[HIP] Device: %s (%d CUs, %d MB)\n", prop.name,
           prop.multiProcessorCount, (int)(prop.totalGlobalMem / 1048576));

    size_t gtable_bytes = secp256k1::G_TABLE_ULONGS * sizeof(uint64_t);
    hipMalloc(&impl_->d_gtable, gtable_bytes);
    hipMalloc(&impl_->d_target, 20);
    hipMalloc(&impl_->d_match_lo, sizeof(uint64_t));
    hipMalloc(&impl_->d_match_hi, sizeof(uint64_t));
    hipMalloc(&impl_->d_match_found, sizeof(uint32_t));

    std::vector<uint64_t> gtable_host(secp256k1::G_TABLE_ULONGS);
    secp256k1::build_g_table(gtable_host.data());
    hipMemcpy(impl_->d_gtable, gtable_host.data(), gtable_bytes, hipMemcpyHostToDevice);
    return true;
}

std::string HIPSolver::device_name() const { return impl_->device_name_str; }
uint64_t HIPSolver::batch_size() const { return impl_->batch; }
void HIPSolver::set_batch_size(uint64_t bs) { if (bs > 0) impl_->batch = bs; }

bool HIPSolver::set_target(const std::array<uint8_t, 20>& hash160) {
    hipMemcpy(impl_->d_target, hash160.data(), 20, hipMemcpyHostToDevice);
    return true;
}

bool HIPSolver::search_batch(uint64_t start_lo, uint64_t start_hi, uint64_t batch_size,
                             uint64_t& found_lo, uint64_t& found_hi) {
    if (batch_size == 0) batch_size = impl_->batch;
    uint64_t z64 = 0; uint32_t z32 = 0;
    hipMemcpy(impl_->d_match_lo, &z64, 8, hipMemcpyHostToDevice);
    hipMemcpy(impl_->d_match_hi, &z64, 8, hipMemcpyHostToDevice);
    hipMemcpy(impl_->d_match_found, &z32, 4, hipMemcpyHostToDevice);

    int threads = impl_->threads_per_block;
    int blocks = (int)((batch_size + threads - 1) / threads);

    hipLaunchKernelGGL(puzzle_search, dim3(blocks), dim3(threads), 0, 0,
        impl_->d_gtable, impl_->d_target, start_lo, start_hi, batch_size,
        impl_->d_match_lo, impl_->d_match_hi, impl_->d_match_found);
    hipDeviceSynchronize();

    uint32_t found = 0;
    hipMemcpy(&found, impl_->d_match_found, 4, hipMemcpyDeviceToHost);
    if (found) {
        hipMemcpy(&found_lo, impl_->d_match_lo, 8, hipMemcpyDeviceToHost);
        hipMemcpy(&found_hi, impl_->d_match_hi, 8, hipMemcpyDeviceToHost);
        return true;
    }
    return false;
}

double HIPSolver::benchmark(uint64_t num_keys) {
    std::array<uint8_t, 20> dummy; memset(dummy.data(), 0xFF, 20);
    set_target(dummy);
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t rem = num_keys, off = 1;
    while (rem > 0) { uint64_t bs = std::min(rem, impl_->batch); uint64_t fl, fh;
        search_batch(off, 0, bs, fl, fh); off += bs; rem -= bs; }
    auto t1 = std::chrono::high_resolution_clock::now();
    return (double)num_keys / std::chrono::duration<double>(t1-t0).count();
}

#endif // USE_HIP
