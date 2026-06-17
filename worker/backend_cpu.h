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
static constexpr int GROUP_H = 256;         // Symmetric half-width: group spans center +/- H
static constexpr int GROUP_SIZE = 2 * GROUP_H + 1;

class CPUBackend : public ComputeBackend {
public:
    CPUBackend(int threads = 0) {
        num_threads_ = threads > 0 ? threads :
            (int)std::thread::hardware_concurrency();
    }

    bool init() override {
        g_table_ = new uint64_t[secp256k1::G_TABLE_ULONGS];
        secp256k1::build_g_table(g_table_);
        build_group_tables();
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

    // Precomputed affine multiples for symmetric group addition:
    //   ig_x_[i], ig_y_[i] = (i*G) affine, for i = 1..GROUP_H
    //   step_x_/step_y_    = (GROUP_SIZE * G) affine, to advance the center
    uint64_t ig_x_[GROUP_H + 1][4];
    uint64_t ig_y_[GROUP_H + 1][4];
    uint64_t step_x_[4], step_y_[4];

    static void affinize_pt(const secp256k1::JacobianPoint& P, uint64_t ax[4], uint64_t ay[4]) {
        uint64_t zi[4], zi2[4], zi3[4];
        secp256k1::mod_inv(zi, P.Z);
        secp256k1::mod_sqr(zi2, zi);
        secp256k1::mod_mul(zi3, zi2, zi);
        secp256k1::mod_mul(ax, P.X, zi2);
        secp256k1::mod_mul(ay, P.Y, zi3);
    }

    void build_group_tables() {
        for (int i = 1; i <= GROUP_H; i++) {
            uint64_t k[4] = {(uint64_t)i, 0, 0, 0};
            secp256k1::JacobianPoint P;
            secp256k1::scalar_mul_g_windowed(P, k, g_table_);
            affinize_pt(P, ig_x_[i], ig_y_[i]);
        }
        uint64_t ks[4] = {(uint64_t)GROUP_SIZE, 0, 0, 0};
        secp256k1::JacobianPoint S;
        secp256k1::scalar_mul_g_windowed(S, ks, g_table_);
        affinize_pt(S, step_x_, step_y_);
    }

    // Incremental search with batch inversion.
    // For each batch of BATCH_INV_SIZE keys:
    //   1. Compute P[0] = (start + offset) * G via windowed table
    //   2. P[i] = P[i-1] + G for i=1..BATCH_INV_SIZE-1 (incremental!)
    //   3. Batch-invert all Z values (Montgomery trick)
    //   4. Affinize, compress, hash, compare
    // External stop signal (set by SIGINT handler)
    static inline std::atomic<bool>* g_stop_flag = nullptr;
public:
    void set_stop_flag(std::atomic<bool>* flag) { g_stop_flag = flag; }
private:

    // Symmetric group-addition search.
    //
    // A "group" of GROUP_SIZE = 2*H+1 consecutive keys is centered on a scalar
    // C. Because the affine multiples i*G are precomputed constants, the points
    // C +/- i*G are obtained from the *single* slope (yi - Cy)/(xi - Cx); the
    // +i and -i points share the same x-difference, so one modular inverse per i
    // serves both. With one batch inversion across the whole group, each output
    // point costs ~4 field muls and is produced already in affine form (no
    // separate affinize pass). The center is advanced C += GROUP_SIZE*G between
    // groups, again via the shared batch inverse.
    void search_incremental(uint64_t base_lo, uint64_t base_hi,
                            uint64_t offset, uint64_t count,
                            const uint8_t target[20],
                            std::atomic<bool>& found,
                            std::atomic<uint64_t>& result_lo,
                            std::atomic<uint64_t>& result_hi) {
        using secp256k1::mod_mul; using secp256k1::mod_sqr; using secp256k1::mod_sub;

        // Emit/compare a found key by its absolute index relative to base.
        auto report_if_match = [&](const uint8_t h160[20], uint64_t key_index) -> bool {
            if (memcmp(h160, target, 20) != 0) return false;
            uint64_t c = 0;
            uint64_t klo = secp256k1::addc(base_lo, key_index, c);
            uint64_t khi = secp256k1::addc(base_hi, 0, c);
            found.store(true, std::memory_order_relaxed);
            result_lo.store(klo);
            result_hi.store(khi);
            return true;
        };

        // Affine point -> compressed pubkey bytes.
        auto write_pubkey = [](const uint64_t ax[4], const uint64_t ay[4], uint8_t pk[33]) {
            pk[0] = 0x02 | (uint8_t)(ay[0] & 1);
            for (int li = 0; li < 4; li++) {
                uint64_t l = ax[3 - li];
                for (int j = 0; j < 8; j++) pk[1 + li * 8 + j] = (uint8_t)(l >> (56 - 8 * j));
            }
        };

        uint64_t ax[GROUP_SIZE][4], ay[GROUP_SIZE][4];

        uint64_t Cx[4], Cy[4];
        bool center_valid = false;

        for (uint64_t g_start = offset; g_start < offset + count;) {
            if (found.load(std::memory_order_relaxed)) return;
            if (g_stop_flag && !g_stop_flag->load(std::memory_order_relaxed)) return;

            // Center scalar index (relative to base) = g_start + GROUP_H.
            uint64_t center_index = g_start + (uint64_t)GROUP_H;

            if (!center_valid) {
                uint64_t k[4]; uint64_t c = 0;
                k[0] = secp256k1::addc(base_lo, center_index, c);
                k[1] = secp256k1::addc(base_hi, 0, c);
                k[2] = c; k[3] = 0;
                secp256k1::JacobianPoint C;
                secp256k1::scalar_mul_g_windowed(C, k, g_table_);
                affinize_pt(C, Cx, Cy);
            }

            // Denominators: den[i] = ig_x[i] - Cx (i=1..H); den[0] = step_x - Cx.
            uint64_t den[GROUP_H + 1][4];
            mod_sub(den[0], step_x_, Cx);
            bool degenerate = (den[0][0] | den[0][1] | den[0][2] | den[0][3]) == 0;
            for (int i = 1; i <= GROUP_H; i++) {
                mod_sub(den[i], ig_x_[i], Cx);
                if ((den[i][0] | den[i][1] | den[i][2] | den[i][3]) == 0) degenerate = true;
            }

            if (degenerate) {
                // Astronomically rare (center.x collides with a table x). Fall back
                // to a direct scalar recompute for this group so we never emit or
                // skip a key, then force a fresh center next group.
                uint64_t span = std::min((uint64_t)GROUP_SIZE, offset + count - g_start);
                for (uint64_t m = 0; m < span; m++) {
                    if (found.load(std::memory_order_relaxed)) return;
                    uint64_t idx = g_start + m;
                    uint64_t k[4]; uint64_t c = 0;
                    k[0] = secp256k1::addc(base_lo, idx, c);
                    k[1] = secp256k1::addc(base_hi, 0, c);
                    k[2] = c; k[3] = 0;
                    secp256k1::JacobianPoint P;
                    secp256k1::scalar_mul_g_windowed(P, k, g_table_);
                    if (secp256k1::is_infinity(P)) continue;
                    uint64_t px[4], py[4]; affinize_pt(P, px, py);
                    uint8_t pk[33], h160[20];
                    write_pubkey(px, py, pk);
                    hash::pubkey_to_hash160(pk, h160);
                    if (report_if_match(h160, idx)) return;
                }
                center_valid = false;
                g_start += GROUP_SIZE;
                continue;
            }

            // Batch invert den[0..H] (Montgomery trick).
            uint64_t pre[GROUP_H + 1][4];
            memcpy(pre[0], den[0], 32);
            for (int i = 1; i <= GROUP_H; i++) mod_mul(pre[i], pre[i - 1], den[i]);
            uint64_t acc[4];
            secp256k1::mod_inv(acc, pre[GROUP_H]);
            uint64_t inv[GROUP_H + 1][4];
            for (int i = GROUP_H; i > 0; i--) {
                mod_mul(inv[i], acc, pre[i - 1]);
                mod_mul(acc, acc, den[i]);
            }
            memcpy(inv[0], acc, 32);

            // Produce all affine points of the group. m maps to scalar index
            // g_start + m; delta = m - GROUP_H selects center (0), +i, or -i.
            memcpy(ax[GROUP_H], Cx, 32);
            memcpy(ay[GROUP_H], Cy, 32);
            for (int i = 1; i <= GROUP_H; i++) {
                // C + i*G  -> index GROUP_H + i
                uint64_t dy[4], s[4], x3[4], y3[4], t[4];
                mod_sub(dy, ig_y_[i], Cy);
                mod_mul(s, dy, inv[i]);
                mod_sqr(x3, s); mod_sub(x3, x3, Cx); mod_sub(x3, x3, ig_x_[i]);
                mod_sub(t, Cx, x3); mod_mul(y3, s, t); mod_sub(y3, y3, Cy);
                memcpy(ax[GROUP_H + i], x3, 32);
                memcpy(ay[GROUP_H + i], y3, 32);
                // C - i*G  -> index GROUP_H - i  (Q = (ig_x[i], -ig_y[i]))
                uint64_t ny[4], dy2[4], s2[4], x3b[4], y3b[4];
                mod_sub(ny, secp256k1::P, ig_y_[i]);   // -ig_y mod p (ig_y != 0)
                mod_sub(dy2, ny, Cy);
                mod_mul(s2, dy2, inv[i]);
                mod_sqr(x3b, s2); mod_sub(x3b, x3b, Cx); mod_sub(x3b, x3b, ig_x_[i]);
                mod_sub(t, Cx, x3b); mod_mul(y3b, s2, t); mod_sub(y3b, y3b, Cy);
                memcpy(ax[GROUP_H - i], x3b, 32);
                memcpy(ay[GROUP_H - i], y3b, 32);
            }

            // Hash + compare over the valid index window [g_start, offset+count).
            uint64_t span = std::min((uint64_t)GROUP_SIZE, offset + count - g_start);
            uint64_t m = 0;
#ifdef HASH_HAVE_AVX2_RMD
            uint8_t pubkeys[8][33];
            for (; m + 8 <= span; m += 8) {
                if (found.load(std::memory_order_relaxed)) return;
                for (int k = 0; k < 8; k++)
                    write_pubkey(ax[m + k], ay[m + k], pubkeys[k]);
                uint8_t h160s[8][20];
                hash::pubkey_to_hash160_8way(pubkeys, h160s);
                for (int k = 0; k < 8; k++)
                    if (report_if_match(h160s[k], g_start + m + k)) return;
            }
#endif
            for (; m < span; m++) {
                uint8_t pk[33], h160[20];
                write_pubkey(ax[m], ay[m], pk);
                hash::pubkey_to_hash160(pk, h160);
                if (report_if_match(h160, g_start + m)) return;
            }

            // Advance center: C = C + GROUP_SIZE*G using the shared inverse inv[0].
            {
                uint64_t dy[4], s[4], nx[4], nyc[4], t[4];
                mod_sub(dy, step_y_, Cy);
                mod_mul(s, dy, inv[0]);
                mod_sqr(nx, s); mod_sub(nx, nx, Cx); mod_sub(nx, nx, step_x_);
                mod_sub(t, Cx, nx); mod_mul(nyc, s, t); mod_sub(nyc, nyc, Cy);
                memcpy(Cx, nx, 32); memcpy(Cy, nyc, 32);
                center_valid = true;
            }
            g_start += GROUP_SIZE;
        }
    }
};
