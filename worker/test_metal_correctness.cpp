// Correctness test for the Metal backend.
//
// Two layers:
//   (A) hash160 reference  — CPU computes h160 for known keys (golden values).
//   (B) range search       — Metal must FIND a known key placed at a non-zero
//                            offset inside a search batch, and return its exact
//                            private key. This exercises the per-thread key
//                            derivation (k = base + gid) and any incremental /
//                            batched-inverse scheme. Run BEFORE and AFTER any
//                            kernel change; output must be identical.
#include "kernels/secp256k1.h"
#include "kernels/hash.h"
#include "kernels/metal_solver.h"
#include <cstdio>
#include <cstring>
#include <array>

static uint64_t* g_gtable = nullptr;

static void compute_h160(uint64_t lo, uint64_t hi, uint8_t out[20]) {
    uint64_t k[4] = {lo, hi, 0, 0};
    secp256k1::JacobianPoint P;
    secp256k1::scalar_mul_g_windowed(P, k, g_gtable);
    if (secp256k1::is_infinity(P)) { memset(out, 0, 20); return; }
    uint64_t inv_z[4], zi2[4], zi3[4], ax[4], ay[4];
    secp256k1::mod_inv(inv_z, P.Z);
    secp256k1::mod_sqr(zi2, inv_z);
    secp256k1::mod_mul(zi3, zi2, inv_z);
    secp256k1::mod_mul(ax, P.X, zi2);
    secp256k1::mod_mul(ay, P.Y, zi3);
    uint8_t pk[33];
    pk[0] = 0x02 | (uint8_t)(ay[0] & 1);
    for (int li = 0; li < 4; li++) {
        uint64_t l = ax[3 - li];
        for (int j = 0; j < 8; j++) pk[1 + li * 8 + j] = (uint8_t)(l >> (56 - 8 * j));
    }
    hash::pubkey_to_hash160(pk, out);
}

struct KeyCase { uint64_t lo, hi; const char* name; };

int main() {
    g_gtable = new uint64_t[secp256k1::G_TABLE_ULONGS];
    secp256k1::build_g_table(g_gtable);

    KeyCase keys[] = {
        {1, 0, "puzzle1"},
        {3, 0, "puzzle2"},
        {0xd2c55, 0, "puzzle20"},
        {0x3d94cd64, 0, "puzzle30"},
        {0xb862a62e, 0, "puzzle32"},
        {0x4aed21170ULL, 0, "puzzle35"},
        {0x1757756a93ULL, 0, "puzzle40"},
        {0x22382facd0ULL, 0, "puzzle42"},
    };

    printf("=== (A) CPU hash160 reference ===\n");
    for (auto& k : keys) {
        uint8_t h[20];
        compute_h160(k.lo, k.hi, h);
        printf("%-9s key=%llx h160=", k.name, (unsigned long long)k.lo);
        for (int i = 0; i < 20; i++) printf("%02x", h[i]);
        printf("\n");
    }

    MetalSolver m;
    if (!m.init()) {
        printf("Metal init failed: %s\n", m.error().c_str());
        delete[] g_gtable;
        return 1;
    }

    int pass = 0, fail = 0;

    // (B) Range search: place each known key at a non-zero offset inside a
    // batch and require Metal to locate the EXACT private key. This catches
    // off-by-one and per-thread-key-derivation bugs that a single-key verify
    // (offset 0) would miss.
    printf("\n=== (B) Metal range search (key at offset>0 in batch) ===\n");
    const uint64_t offsets[] = {0, 1, 37, 1000, 65535, 1000000};
    for (auto& k : keys) {
        uint8_t h[20];
        compute_h160(k.lo, k.hi, h);
        std::array<uint8_t, 20> tgt;
        memcpy(tgt.data(), h, 20);
        m.set_target(tgt);

        bool all_ok = true;
        for (uint64_t off : offsets) {
            if (off > k.lo) continue;            // can't start below 0
            uint64_t start_lo = k.lo - off;
            // batch must cover the key: span = off + margin
            uint64_t span = off + 1 + 50;
            auto res = m.search_batch(start_lo, 0, span);
            bool ok = res.found &&
                      res.private_key.d[0] == k.lo &&
                      res.private_key.d[1] == k.hi;
            if (!ok) {
                all_ok = false;
                printf("  %-9s off=%-8llu FAIL (found=%d lo=%llx)\n",
                       k.name, (unsigned long long)off, res.found,
                       (unsigned long long)res.private_key.d[0]);
            }
        }
        printf("  %-9s range-search %s\n", k.name, all_ok ? "PASS" : "FAIL");
        all_ok ? pass++ : fail++;
    }

    // (C) No false positives: search a batch that does NOT contain the key.
    printf("\n=== (C) Metal no-false-positive ===\n");
    {
        uint8_t h[20];
        compute_h160(0x4aed21170ULL, 0, h);     // puzzle35 target
        std::array<uint8_t, 20> tgt; memcpy(tgt.data(), h, 20);
        m.set_target(tgt);
        // search a window far from the real key
        auto res = m.search_batch(0x1000000ULL, 0, 1000000);
        bool ok = !res.found;
        printf("  absent-key search %s\n", ok ? "PASS (not found)" : "FAIL (false positive)");
        ok ? pass++ : fail++;
    }

    printf("\n=== Result: %d passed, %d failed ===\n", pass, fail);
    delete[] g_gtable;
    return fail > 0 ? 1 : 0;
}
