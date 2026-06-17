// Single-thread throughput benchmark for the CPU search inner loop.
// Mirrors backend_cpu.h's search_incremental (incremental add + batch inversion).
#include "kernels/secp256k1.h"
#include "kernels/hash.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <chrono>

static constexpr int BATCH = 256;

int main(int argc, char** argv) {
    uint64_t N = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 2000000;
    uint64_t* g_table = new uint64_t[secp256k1::G_TABLE_ULONGS];
    secp256k1::build_g_table(g_table);

    const uint64_t* GX = secp256k1::GX;
    const uint64_t* GY = secp256k1::GY;
    uint8_t target[20]; memset(target, 0xAB, 20);
    uint64_t base_lo = 0x123456789ull, base_hi = 0;
    volatile uint64_t sink = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t processed = 0;
    while (processed < N) {
        int bs = (int)std::min((uint64_t)BATCH, N - processed);
        uint64_t k0[4] = { base_lo + processed, base_hi, 0, 0 };
        secp256k1::JacobianPoint pts[BATCH];
        secp256k1::scalar_mul_g_windowed(pts[0], k0, g_table);
        for (int i = 1; i < bs; i++) secp256k1::point_add_mixed(pts[i], pts[i-1], GX, GY);

        uint64_t prod[BATCH][4];
        memcpy(prod[0], pts[0].Z, 32);
        for (int i = 1; i < bs; i++) secp256k1::mod_mul(prod[i], prod[i-1], pts[i].Z);
        uint64_t inv[4]; secp256k1::mod_inv(inv, prod[bs-1]);
        uint64_t zinv[BATCH][4];
        for (int i = bs-1; i > 0; i--) {
            secp256k1::mod_mul(zinv[i], inv, prod[i-1]);
            secp256k1::mod_mul(inv, inv, pts[i].Z);
        }
        memcpy(zinv[0], inv, 32);

        int i = 0;
#ifdef HASH_HAVE_AVX2_RMD
        uint8_t pubkeys[8][33];
        for (; i + 8 <= bs; i += 8) {
            for (int k = 0; k < 8; k++) {
                uint64_t zi2[4], zi3[4], ax[4], ay[4];
                secp256k1::mod_sqr(zi2, zinv[i+k]);
                secp256k1::mod_mul(zi3, zi2, zinv[i+k]);
                secp256k1::mod_mul(ax, pts[i+k].X, zi2);
                secp256k1::mod_mul(ay, pts[i+k].Y, zi3);
                pubkeys[k][0] = 0x02 | (uint8_t)(ay[0] & 1);
                for (int li = 0; li < 4; li++) {
                    uint64_t l = ax[3-li];
                    for (int j = 0; j < 8; j++) pubkeys[k][1+li*8+j] = (uint8_t)(l >> (56-8*j));
                }
            }
            uint8_t h160s[8][20];
            hash::pubkey_to_hash160_8way(pubkeys, h160s);
            for (int k = 0; k < 8; k++) sink += h160s[k][0];
        }
#endif
        for (; i < bs; i++) {
            uint64_t zi2[4], zi3[4], ax[4], ay[4];
            secp256k1::mod_sqr(zi2, zinv[i]);
            secp256k1::mod_mul(zi3, zi2, zinv[i]);
            secp256k1::mod_mul(ax, pts[i].X, zi2);
            secp256k1::mod_mul(ay, pts[i].Y, zi3);
            uint8_t pubkey[33];
            pubkey[0] = 0x02 | (uint8_t)(ay[0] & 1);
            for (int li = 0; li < 4; li++) {
                uint64_t l = ax[3-li];
                for (int j = 0; j < 8; j++) pubkey[1+li*8+j] = (uint8_t)(l >> (56-8*j));
            }
            uint8_t h160[20];
            hash::pubkey_to_hash160(pubkey, h160);
            sink += h160[0];
        }
        processed += bs;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double el = std::chrono::duration<double>(t1 - t0).count();
    printf("keys=%llu  time=%.3fs  rate=%.3f MK/s  (sink=%llu)\n",
           (unsigned long long)N, el, N / el / 1e6, (unsigned long long)sink);
    delete[] g_table;
    return 0;
}
