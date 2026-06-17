#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_solver.h"
#include "secp256k1.h"
#include <chrono>
#include <cstring>
#include <vector>

// Embedded Metal shader source
static const char* kMetalShaderSource =
#include "puzzle_metal_source.inc"
;

static constexpr uint32_t kThreadgroupWidth = 32;
static constexpr uint64_t kDefaultBatchSize = 4'000'000ull;
// Symmetric group-addition: each GPU thread owns one group of GROUP_SIZE =
// 2*GROUP_H+1 consecutive keys centered on C, sharing ONE mod_inv across the
// group (Montgomery batch inversion). Must match GROUP_H in puzzle.metal.
static constexpr uint32_t kGroupH = 256;
static constexpr uint64_t kKeysPerThread = 2 * kGroupH + 1;

struct MetalSolver::Impl {
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLLibrary>              lib      = nil;
    id<MTLComputePipelineState> pipe     = nil;

    id<MTLBuffer> b_gtable    = nil;
    id<MTLBuffer> b_igtable   = nil;   // i*G affine table, i=1..GROUP_H (+ step at slot 0)
    id<MTLBuffer> b_target    = nil;
    id<MTLBuffer> b_match_lo  = nil;
    id<MTLBuffer> b_match_hi  = nil;
    id<MTLBuffer> b_match_fnd = nil;

    uint64_t batch = kDefaultBatchSize;
    std::string device_name_str;
    std::string error_str;
};

MetalSolver::MetalSolver() : impl_(new Impl{}) {}

MetalSolver::~MetalSolver() {
    @autoreleasepool {
        delete impl_;
    }
}

std::string MetalSolver::device_name() const { return impl_->device_name_str; }
std::string MetalSolver::error() const { return impl_->error_str; }
uint64_t MetalSolver::batch_size() const { return impl_->batch; }

void MetalSolver::set_batch_size(uint64_t bs) {
    if (bs == 0) bs = kDefaultBatchSize;
    impl_->batch = ((bs + kThreadgroupWidth - 1) / kThreadgroupWidth) * kThreadgroupWidth;
}

bool MetalSolver::init() {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            impl_->error_str = "No Metal device found";
            return false;
        }
        impl_->device_name_str = std::string([impl_->device.name UTF8String]);

        NSString* source = [NSString stringWithUTF8String:kMetalShaderSource];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion2_4;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        } else {
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = YES;
            #pragma clang diagnostic pop
        }
#else
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
        #pragma clang diagnostic pop
#endif

        NSError* err = nil;
        impl_->lib = [impl_->device newLibraryWithSource:source options:opts error:&err];
        if (!impl_->lib) {
            impl_->error_str = std::string("MSL compile failed: ")
                + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }

        id<MTLFunction> fn = [impl_->lib newFunctionWithName:@"puzzle_search"];
        if (!fn) {
            impl_->error_str = "puzzle_search kernel not found";
            return false;
        }
        impl_->pipe = [impl_->device newComputePipelineStateWithFunction:fn error:&err];
        if (!impl_->pipe) {
            impl_->error_str = "Pipeline creation failed";
            return false;
        }

        impl_->queue = [impl_->device newCommandQueue];

        // Allocate buffers
        static constexpr size_t gtable_bytes = secp256k1::G_TABLE_ULONGS * sizeof(uint64_t);
        impl_->b_gtable    = [impl_->device newBufferWithLength:gtable_bytes
                                                        options:MTLResourceStorageModeShared];
        // i*G affine table: slot 0 = step (GROUP_SIZE*G), slots 1..H = i*G.
        static constexpr size_t igtable_slots = (size_t)(kGroupH + 1);
        static constexpr size_t igtable_bytes = igtable_slots * 8 * sizeof(uint64_t);
        impl_->b_igtable   = [impl_->device newBufferWithLength:igtable_bytes
                                                        options:MTLResourceStorageModeShared];
        impl_->b_target    = [impl_->device newBufferWithLength:20
                                                        options:MTLResourceStorageModeShared];
        impl_->b_match_lo  = [impl_->device newBufferWithLength:sizeof(uint64_t)
                                                        options:MTLResourceStorageModeShared];
        impl_->b_match_hi  = [impl_->device newBufferWithLength:sizeof(uint64_t)
                                                        options:MTLResourceStorageModeShared];
        impl_->b_match_fnd = [impl_->device newBufferWithLength:sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];

        if (!impl_->b_gtable || !impl_->b_igtable || !impl_->b_target ||
            !impl_->b_match_lo || !impl_->b_match_hi || !impl_->b_match_fnd) {
            impl_->error_str = "Buffer allocation failed";
            return false;
        }

        // Build G table on CPU and upload
        std::vector<uint64_t> gtable_host(secp256k1::G_TABLE_ULONGS);
        secp256k1::build_g_table(gtable_host.data());
        memcpy([impl_->b_gtable contents], gtable_host.data(), gtable_bytes);

        // Build i*G affine table: slot 0 = (GROUP_SIZE)*G (center step), slots
        // 1..H = i*G. Each slot is 8 ulongs: x[0..3], y[0..3]. Shared by every
        // thread for symmetric group addition.
        {
            uint64_t* ig = (uint64_t*)[impl_->b_igtable contents];
            auto affine_of = [&](uint64_t scalar, uint64_t* out_x, uint64_t* out_y) {
                uint64_t k[4] = {scalar, 0, 0, 0};
                secp256k1::JacobianPoint P;
                secp256k1::scalar_mul_g_windowed(P, k, gtable_host.data());
                uint64_t zi[4], zi2[4], zi3[4];
                secp256k1::mod_inv(zi, P.Z);
                secp256k1::mod_sqr(zi2, zi);
                secp256k1::mod_mul(zi3, zi2, zi);
                secp256k1::mod_mul(out_x, P.X, zi2);
                secp256k1::mod_mul(out_y, P.Y, zi3);
            };
            affine_of((uint64_t)kKeysPerThread, &ig[0], &ig[4]);          // step
            for (uint32_t i = 1; i <= kGroupH; ++i)
                affine_of((uint64_t)i, &ig[i * 8], &ig[i * 8 + 4]);       // i*G
        }

        return true;
    }
}

bool MetalSolver::set_target(const std::array<uint8_t, 20>& hash160) {
    if (!impl_->b_target) return false;
    memcpy([impl_->b_target contents], hash160.data(), 20);
    return true;
}

SearchResult MetalSolver::search_batch(uint64_t start_lo, uint64_t start_hi,
                                       uint64_t batch_size) {
    SearchResult result{};
    if (!impl_->pipe) return result;
    if (batch_size == 0) batch_size = impl_->batch;

    // Reset match slots
    *(uint64_t*)[impl_->b_match_lo contents] = 0;
    *(uint64_t*)[impl_->b_match_hi contents] = 0;
    *(uint32_t*)[impl_->b_match_fnd contents] = 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:impl_->pipe];

        uint64_t total_keys = batch_size;
        [enc setBuffer:impl_->b_gtable    offset:0 atIndex:0];
        [enc setBuffer:impl_->b_target    offset:0 atIndex:1];
        [enc setBytes:&start_lo   length:sizeof(start_lo)   atIndex:2];
        [enc setBytes:&start_hi   length:sizeof(start_hi)   atIndex:3];
        [enc setBytes:&total_keys length:sizeof(total_keys) atIndex:4];
        [enc setBuffer:impl_->b_match_lo  offset:0 atIndex:5];
        [enc setBuffer:impl_->b_match_hi  offset:0 atIndex:6];
        [enc setBuffer:impl_->b_match_fnd offset:0 atIndex:7];
        [enc setBuffer:impl_->b_igtable   offset:0 atIndex:8];

        NSUInteger maxTPT = [impl_->pipe maxTotalThreadsPerThreadgroup];
        NSUInteger tg = (kThreadgroupWidth <= maxTPT) ? kThreadgroupWidth : maxTPT;

        // Each thread processes kKeysPerThread consecutive keys.
        uint64_t num_threads = (total_keys + kKeysPerThread - 1) / kKeysPerThread;

        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    uint32_t found = *(const uint32_t*)[impl_->b_match_fnd contents];
    if (found != 0) {
        result.found = true;
        result.private_key.d[0] = *(const uint64_t*)[impl_->b_match_lo contents];
        result.private_key.d[1] = *(const uint64_t*)[impl_->b_match_hi contents];
    }
    return result;
}

BenchmarkResult MetalSolver::benchmark(uint64_t num_keys) {
    // Set a dummy target that won't match
    std::array<uint8_t, 20> dummy{};
    memset(dummy.data(), 0xFF, 20);
    set_target(dummy);

    auto t0 = std::chrono::high_resolution_clock::now();

    uint64_t remaining = num_keys;
    uint64_t offset = 1;  // start from key=1
    while (remaining > 0) {
        uint64_t batch = std::min(remaining, impl_->batch);
        search_batch(offset, 0, batch);
        offset += batch;
        remaining -= batch;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    BenchmarkResult br;
    br.keys_per_second = (double)num_keys / elapsed;
    br.elapsed_seconds = elapsed;
    br.keys_checked = num_keys;
    br.device_name = "Metal GPU: " + impl_->device_name_str;
    return br;
}

bool MetalSolver::verify_key(uint64_t priv_lo, uint64_t priv_hi,
                             const std::array<uint8_t, 20>& expected_h160) {
    set_target(expected_h160);
    auto res = search_batch(priv_lo, priv_hi, 1);
    return res.found;
}
