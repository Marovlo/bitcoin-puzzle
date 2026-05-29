#pragma once
#include "worker.h"
#include "kernels/secp256k1.h"
#include "kernels/hash.h"
#include <chrono>
#include <cstring>
#include <thread>
#include <atomic>
#include <vector>

// P0 Optimizations:
// 1. Incremental search: P[i+1] = P[i] + G (1 point_add vs 32 per key)
// 2. Batch inversion (Montgomery trick): N keys share 1 mod_inv + 3N mod_mul
//
// Pipeline per batch of BATCH_SIZE keys:
//   a) Compute all Jacobian points: P[0] via table, P[i]=P[i-1]+G
//   b) Batch-invert all Z coordinates (1 mod_inv for entire batch)
//   c) Affinize all points, compress, hash, compare

static constexpr int BATCH_INV_SIZE = 256; // Keys per batch inversion

class CPUBackend : public ComputeBackend {
public:
    CPUBackend(int threads = 0) {
        num_threads_ = threads > 0 ? threads :
            (int)std::thread::hardware_concurrency();
    }

    bool init() override {
        g_table_ = new uint64_t[secp256k1::G_TABLE_ULONGS];
        secp256k1::build_g_table(g_table_);
        return true;
    }

    std::string name() const override {
        return "cpu_" + std::to_string(num_threads_) + "t";
    }

    bool search(uint64_t start_lo, uint64_t start_hi, uint64_t size,
                const uint8_t target_h160[20],
                uint64_t& found_lo, uint64_t& found_hi) override {
        std::atomic<bool> found{false};
        std::atomic<uint64_t> result_lo{0}, result_hi{0};
        uint8_t target_copy[20];
        memcpy(target_copy, target_h160, 20);

        uint64_t chunk_per_thread = size / num_threads_;
        std::vector<std::thread> threads;

        for (int t = 0; t < num_threads_; t++) {
            uint64_t t_offset = (uint64_t)t * chunk_per_thread;
            uint64_t t_size = (t == num_threads_ - 1) ? (size - t_offset) : chunk_per_thread;

            threads.emplace_back([&, t_offset, t_size]() {
                search_incremental(start_lo, start_hi, t_offset, t_size,
                                   target_copy, found, result_lo, result_hi);
            });
        }
        for (auto& th : threads) th.join();

        if (found.load()) {
            found_lo = result_lo.load();
            found_hi = result_hi.load();
            return true;
        }
        return false;
    }

    uint64_t benchmark(uint64_t sample_size) override {
        uint8_t dummy[20]; memset(dummy, 0xFF, 20);
        uint64_t fl, fh;
        auto t0 = std::chrono::high_resolution_clock::now();
        search(1, 0, sample_size, dummy, fl, fh);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        return (uint64_t)((double)sample_size / elapsed);
    }

    ~CPUBackend() override { delete[] g_table_; }

private:
    uint64_t* g_table_ = nullptr;
    int num_threads_;

    // Incremental search with batch inversion.
    // For each batch of BATCH_INV_SIZE keys:
    //   1. Compute P[0] = (start + offset) * G via windowed table
    //   2. P[i] = P[i-1] + G for i=1..BATCH_INV_SIZE-1 (incremental!)
    //   3. Batch-invert all Z values (Montgomery trick)
    //   4. Affinize, compress, hash, compare
    void search_incremental(uint64_t base_lo, uint64_t base_hi,
                            uint64_t offset, uint64_t count,
                            const uint8_t target[20],
                            std::atomic<bool>& found,
                            std::atomic<uint64_t>& result_lo,
                            std::atomic<uint64_t>& result_hi) {
        // Precompute G in affine for incremental addition
        static const uint64_t* GX = secp256k1::GX;
        static const uint64_t* GY = secp256k1::GY;

        uint64_t processed = 0;
        while (processed < count && !found.load(std::memory_order_relaxed)) {
            int batch_size = (int)std::min((uint64_t)BATCH_INV_SIZE, count - processed);

            // Compute first key in batch via full scalar mul
            uint64_t first_idx = offset + processed;
            uint64_t k0[4];
            { uint64_t c = 0;
              k0[0] = secp256k1::addc(base_lo, first_idx, c);
              k0[1] = secp256k1::addc(base_hi, 0, c);
              k0[2] = c; k0[3] = 0; }

            secp256k1::JacobianPoint P0;
            secp256k1::scalar_mul_g_windowed(P0, k0, g_table_);

            // Store all Jacobian points for this batch
            // Use stack array for small batch
            secp256k1::JacobianPoint points[BATCH_INV_SIZE];
            points[0] = P0;

            // Incremental: P[i] = P[i-1] + G
            for (int i = 1; i < batch_size; i++) {
                secp256k1::point_add_mixed(points[i], points[i-1], GX, GY);
            }

            // Batch inversion (Montgomery trick):
            // Given Z[0..n-1], compute inv(Z[i]) for all i using only 1 mod_inv.
            // 1. products[i] = Z[0] * Z[1] * ... * Z[i]
            // 2. inv_all = mod_inv(products[n-1])
            // 3. Back-propagate: inv(Z[i]) = inv_all * products[i-1]
            uint64_t products[BATCH_INV_SIZE][4];
            memcpy(products[0], points[0].Z, 32);
            for (int i = 1; i < batch_size; i++) {
                secp256k1::mod_mul(products[i], products[i-1], points[i].Z);
            }

            // One expensive mod_inv for the entire batch
            uint64_t inv_all[4];
            secp256k1::mod_inv(inv_all, products[batch_size - 1]);

            // Back-propagate to get individual Z inverses
            uint64_t z_invs[BATCH_INV_SIZE][4];
            for (int i = batch_size - 1; i > 0; i--) {
                secp256k1::mod_mul(z_invs[i], inv_all, products[i-1]);
                secp256k1::mod_mul(inv_all, inv_all, points[i].Z);
            }
            memcpy(z_invs[0], inv_all, 32);

            // Affinize + hash + compare
            for (int i = 0; i < batch_size; i++) {
                if (found.load(std::memory_order_relaxed)) return;

                if (secp256k1::is_infinity(points[i])) continue;

                uint64_t zi2[4], zi3[4], ax[4], ay[4];
                secp256k1::mod_sqr(zi2, z_invs[i]);
                secp256k1::mod_mul(zi3, zi2, z_invs[i]);
                secp256k1::mod_mul(ax, points[i].X, zi2);
                secp256k1::mod_mul(ay, points[i].Y, zi3);

                uint8_t pubkey[33];
                pubkey[0] = 0x02 | (uint8_t)(ay[0] & 1);
                for (int li = 0; li < 4; li++) {
                    uint64_t l = ax[3 - li];
                    for (int j = 0; j < 8; j++)
                        pubkey[1 + li * 8 + j] = (uint8_t)(l >> (56 - 8 * j));
                }

                uint8_t h160[20];
                hash::pubkey_to_hash160(pubkey, h160);

                if (memcmp(h160, target, 20) == 0) {
                    // Reconstruct key = base + offset + processed + i
                    uint64_t key_idx = first_idx + (uint64_t)i;
                    uint64_t c = 0;
                    uint64_t klo = secp256k1::addc(base_lo, key_idx, c);
                    uint64_t khi = secp256k1::addc(base_hi, 0, c);
                    found.store(true, std::memory_order_relaxed);
                    result_lo.store(klo);
                    result_hi.store(khi);
                    return;
                }
            }
            processed += batch_size;
        }
    }
};
