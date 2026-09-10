// Correctness harness for the OpenCL (GPU) backend.
//
// Strategy: generate the expected hash160 on the CPU using the same secp256k1 +
// hash code the CPU backend uses, then ask the GPU to find that hash160 inside a
// known range and check the exact offset it reports. This validates the whole
// GPU pipeline (scalar mul, affine conversion, pubkey encoding, SHA-256,
// RIPEMD-160, range indexing) against an independent implementation.
//
// Build: see build_windows.ps1. Run from the worker/ directory.
#ifdef USE_OPENCL

#include "kernels/opencl/opencl_solver.h"
#include "kernels/secp256k1.h"
#include "kernels/hash.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

// hash160 of the compressed pubkey for scalar k, computed on the CPU.
void cpu_h160(const uint64_t k[4], const uint64_t* g_table, uint8_t h160[20]) {
    secp256k1::JacobianPoint P;
    secp256k1::scalar_mul_g_windowed(P, k, g_table);

    uint64_t zinv[4], zi2[4], zi3[4], ax[4], ay[4];
    secp256k1::mod_inv(zinv, P.Z);
    secp256k1::mod_sqr(zi2, zinv);
    secp256k1::mod_mul(zi3, zi2, zinv);
    secp256k1::mod_mul(ax, P.X, zi2);
    secp256k1::mod_mul(ay, P.Y, zi3);

    uint8_t pubkey[33];
    pubkey[0] = (uint8_t)(0x02 | (ay[0] & 1));
    for (int li = 0; li < 4; li++) {
        const uint64_t limb = ax[3 - li];
        for (int j = 0; j < 8; j++)
            pubkey[1 + li * 8 + j] = (uint8_t)(limb >> (56 - 8 * j));
    }
    hash::pubkey_to_hash160(pubkey, h160);
}

void check(bool cond, const std::string& what) {
    if (cond) {
        g_pass++;
        printf("  ok   %s\n", what.c_str());
    } else {
        g_fail++;
        printf("  FAIL %s\n", what.c_str());
    }
}

} // namespace

int main() {
    // ---- host reference tables ----
    std::vector<uint64_t> gt(secp256k1::G_TABLE_ULONGS);
    secp256k1::build_g_table(gt.data());

    OpenCLSolver solver;
    if (!solver.init(0)) {
        fprintf(stderr, "OpenCL init failed\n");
        return 2;
    }
    printf("\n=== OpenCL correctness ===\n");
    printf("device: %s, batch %llu\n\n", solver.device_name().c_str(),
           (unsigned long long)solver.batch_size());

    // ------------------------------------------------------------------
    // 1. Every offset in a small range must be found exactly, walking the
    //    range in odd-sized chunks so batch boundaries and tails are covered.
    // ------------------------------------------------------------------
    printf("[1] 64 consecutive offsets, chunked in 7s\n");
    {
        const uint64_t base_lo = 0x1;   // small base, keeps the CPU reference cheap
        const uint64_t base_hi = 0;
        const uint64_t span = 64;

        for (uint64_t j = 0; j < span; j++) {
            uint64_t k[4] = { base_lo + j, base_hi, 0, 0 };
            uint8_t target[20];
            cpu_h160(k, gt.data(), target);

            std::array<uint8_t, 20> tgt;
            memcpy(tgt.data(), target, 20);
            solver.set_target(tgt);

            bool found = false;
            uint64_t fl = 0, fh = 0;
            for (uint64_t off = 0; off < span && !found; ) {
                const uint64_t bs = (span - off < 7) ? (span - off) : 7;
                found = solver.search_batch(base_lo + off, base_hi, bs, fl, fh);
                off += bs;
            }

            char label[128];
            snprintf(label, sizeof(label), "offset %llu", (unsigned long long)j);
            check(found && fl == base_lo + j && fh == base_hi, label);
        }
    }

    // ------------------------------------------------------------------
    // 2. 128-bit carry: scanning across a 2^64 boundary must report the right
    //    high limb. This is where the CUDA/HIP/C++ backends classically break.
    // ------------------------------------------------------------------
    printf("\n[2] scan across the 2^64 boundary\n");
    {
        const uint64_t base_lo = 0xFFFFFFFFFFFFFFF0ull;  // carries into hi after 16 keys
        const uint64_t base_hi = 0x1;
        const uint64_t span = 32;
        const uint64_t hit = 20;                          // past the wrap

        uint64_t k[4] = { 0, 0, 0, 0 };
        k[0] = base_lo + hit;
        k[1] = base_hi + ((k[0] < base_lo) ? 1 : 0);
        uint8_t target[20];
        cpu_h160(k, gt.data(), target);

        std::array<uint8_t, 20> tgt;
        memcpy(tgt.data(), target, 20);
        solver.set_target(tgt);

        uint64_t fl = 0, fh = 0;
        const bool found = solver.search_batch(base_lo, base_hi, span, fl, fh);
        char label[160];
        snprintf(label, sizeof(label), "found %s lo=%llx hi=%llx (expect lo=%llx hi=%llx)",
                 found ? "yes" : "NO", (unsigned long long)fl, (unsigned long long)fh,
                 (unsigned long long)k[0], (unsigned long long)k[1]);
        check(found && fl == k[0] && fh == k[1], label);
    }

    // ------------------------------------------------------------------
    // 3. A real-sized dispatch (multi-million keys) must still land on the
    //    exact key. This is the shape of an actual production batch.
    // ------------------------------------------------------------------
    printf("\n[3] production-sized dispatch\n");
    {
        const uint64_t base_lo = 0x100000000ull;         // 2^32
        const uint64_t base_hi = 0;
        const uint64_t span = 2'000'000ull;
        const uint64_t hit = 1'234'567ull;

        uint64_t k[4] = { base_lo + hit, base_hi, 0, 0 };
        uint8_t target[20];
        cpu_h160(k, gt.data(), target);

        std::array<uint8_t, 20> tgt;
        memcpy(tgt.data(), target, 20);
        solver.set_target(tgt);

        uint64_t fl = 0, fh = 0;
        const auto t0 = std::chrono::high_resolution_clock::now();
        const bool found = solver.search_batch(base_lo, base_hi, span, fl, fh);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();

        char label[192];
        snprintf(label, sizeof(label), "found %s at lo=%llx in %llu keys (%.1fs, %.1f MK/s)",
                 found ? "yes" : "NO", (unsigned long long)fl,
                 (unsigned long long)span, elapsed, span / elapsed / 1e6);
        check(found && fl == base_lo + hit && fh == base_hi, label);
    }

    // ------------------------------------------------------------------
    // 4. Negative cases: a target that is not in the range must not be
    //    reported, and neither must a garbage target.
    // ------------------------------------------------------------------
    printf("\n[4] negative cases\n");
    {
        const uint64_t base_lo = 0x500000000ull;
        const uint64_t base_hi = 0;

        // Outside the scanned window
        uint64_t k[4] = { base_lo + 100'000ull, base_hi, 0, 0 };
        uint8_t target[20];
        cpu_h160(k, gt.data(), target);
        std::array<uint8_t, 20> tgt;
        memcpy(tgt.data(), target, 20);
        solver.set_target(tgt);

        uint64_t fl = 0, fh = 0;
        const bool found = solver.search_batch(base_lo, base_hi, 1000, fl, fh);
        check(!found, "key outside the window is not reported");

        std::array<uint8_t, 20> junk;
        memset(junk.data(), 0xAB, 20);
        solver.set_target(junk);
        const bool found_junk = solver.search_batch(base_lo, base_hi, 1000, fl, fh);
        check(!found_junk, "garbage target is not reported");
    }

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

#else
#include <cstdio>
int main() {
    fprintf(stderr, "built without -DUSE_OPENCL\n");
    return 2;
}
#endif
