#pragma once
#include "worker.h"
#include "kernels/secp256k1.h"
#include "kernels/hash.h"
#include <chrono>
#include <cstring>
#include <thread>
#include <atomic>
#include <vector>

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
            uint64_t t_start = (uint64_t)t * chunk_per_thread;
            uint64_t t_size = (t == num_threads_ - 1) ? (size - t_start) : chunk_per_thread;

            threads.emplace_back([&, t_start, t_size]() {
                search_range(start_lo, start_hi, t_start, t_size,
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
        uint8_t dummy_target[20];
        memset(dummy_target, 0xFF, 20);
        uint64_t fl, fh;
        auto t0 = std::chrono::high_resolution_clock::now();
        search(1, 0, sample_size, dummy_target, fl, fh);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        return (uint64_t)((double)sample_size / elapsed);
    }

    ~CPUBackend() override { delete[] g_table_; }

private:
    uint64_t* g_table_ = nullptr;
    int num_threads_;

    void search_range(uint64_t base_lo, uint64_t base_hi,
                      uint64_t offset, uint64_t count,
                      const uint8_t target[20],
                      std::atomic<bool>& found,
                      std::atomic<uint64_t>& result_lo,
                      std::atomic<uint64_t>& result_hi) {
        for (uint64_t i = 0; i < count && !found.load(std::memory_order_relaxed); i++) {
            uint64_t idx = offset + i;
            uint64_t k[4];
            uint64_t c = 0;
            k[0] = secp256k1::addc(base_lo, idx, c);
            k[1] = secp256k1::addc(base_hi, 0, c);
            k[2] = c;
            k[3] = 0;

            secp256k1::JacobianPoint P;
            secp256k1::scalar_mul_g_windowed(P, k, g_table_);
            if (secp256k1::is_infinity(P)) continue;

            uint64_t inv_z[4], inv_z2[4], inv_z3[4], ax[4], ay[4];
            secp256k1::mod_inv(inv_z, P.Z);
            secp256k1::mod_sqr(inv_z2, inv_z);
            secp256k1::mod_mul(inv_z3, inv_z2, inv_z);
            secp256k1::mod_mul(ax, P.X, inv_z2);
            secp256k1::mod_mul(ay, P.Y, inv_z3);

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
                found.store(true, std::memory_order_relaxed);
                result_lo.store(k[0]);
                result_hi.store(k[1]);
                return;
            }
        }
    }
};
