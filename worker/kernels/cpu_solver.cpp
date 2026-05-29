#include "cpu_solver.h"
#include "secp256k1.h"
#include "hash.h"
#include <chrono>
#include <cstring>

CPUSolver::CPUSolver() {}

CPUSolver::~CPUSolver() {
    delete[] g_table_;
}

bool CPUSolver::init() {
    g_table_ = new uint64_t[secp256k1::G_TABLE_ULONGS];
    secp256k1::build_g_table(g_table_);
    return true;
}

SearchResult CPUSolver::search_batch(uint64_t start_lo, uint64_t start_hi,
                                     uint64_t batch_size,
                                     const std::array<uint8_t, 20>& target_h160) {
    SearchResult result{};

    for (uint64_t i = 0; i < batch_size; ++i) {
        // k = start + i
        uint64_t k[4];
        uint64_t c = 0;
        k[0] = secp256k1::addc(start_lo, i, c);
        k[1] = secp256k1::addc(start_hi, 0, c);
        k[2] = c;
        k[3] = 0;

        // P = k * G via windowed table
        secp256k1::JacobianPoint P;
        secp256k1::scalar_mul_g_windowed(P, k, g_table_);

        if (secp256k1::is_infinity(P)) continue;

        // Affinize
        uint64_t inv_z[4], inv_z2[4], inv_z3[4], ax[4], ay[4];
        secp256k1::mod_inv(inv_z, P.Z);
        secp256k1::mod_sqr(inv_z2, inv_z);
        secp256k1::mod_mul(inv_z3, inv_z2, inv_z);
        secp256k1::mod_mul(ax, P.X, inv_z2);
        secp256k1::mod_mul(ay, P.Y, inv_z3);

        // Compress pubkey
        uint8_t pubkey[33];
        pubkey[0] = 0x02 | (uint8_t)(ay[0] & 1);
        for (int li = 0; li < 4; ++li) {
            uint64_t l = ax[3 - li];
            for (int j = 0; j < 8; ++j) {
                pubkey[1 + li * 8 + j] = (uint8_t)(l >> (56 - 8 * j));
            }
        }

        // Hash160
        uint8_t h160[20];
        hash::pubkey_to_hash160(pubkey, h160);

        // Compare
        if (memcmp(h160, target_h160.data(), 20) == 0) {
            result.found = true;
            result.private_key.d[0] = k[0];
            result.private_key.d[1] = k[1];
            result.private_key.d[2] = k[2];
            result.private_key.d[3] = k[3];
            return result;
        }
    }
    return result;
}

BenchmarkResult CPUSolver::benchmark(uint64_t num_keys) {
    // Use a dummy target that will never match
    std::array<uint8_t, 20> dummy_target{};
    memset(dummy_target.data(), 0xFF, 20);

    auto t0 = std::chrono::high_resolution_clock::now();
    search_batch(1, 0, num_keys, dummy_target);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    BenchmarkResult br;
    br.keys_per_second = (double)num_keys / elapsed;
    br.elapsed_seconds = elapsed;
    br.keys_checked = num_keys;
    br.device_name = "CPU (Apple Silicon)";
    return br;
}

bool CPUSolver::verify_key(uint64_t priv_lo, uint64_t priv_hi,
                           const std::array<uint8_t, 20>& expected_h160) {
    uint64_t k[4] = {priv_lo, priv_hi, 0, 0};

    secp256k1::JacobianPoint P;
    secp256k1::scalar_mul_g_windowed(P, k, g_table_);

    if (secp256k1::is_infinity(P)) return false;

    uint64_t inv_z[4], inv_z2[4], inv_z3[4], ax[4], ay[4];
    secp256k1::mod_inv(inv_z, P.Z);
    secp256k1::mod_sqr(inv_z2, inv_z);
    secp256k1::mod_mul(inv_z3, inv_z2, inv_z);
    secp256k1::mod_mul(ax, P.X, inv_z2);
    secp256k1::mod_mul(ay, P.Y, inv_z3);

    uint8_t pubkey[33];
    pubkey[0] = 0x02 | (uint8_t)(ay[0] & 1);
    for (int li = 0; li < 4; ++li) {
        uint64_t l = ax[3 - li];
        for (int j = 0; j < 8; ++j) {
            pubkey[1 + li * 8 + j] = (uint8_t)(l >> (56 - 8 * j));
        }
    }

    uint8_t h160[20];
    hash::pubkey_to_hash160(pubkey, h160);
    return memcmp(h160, expected_h160.data(), 20) == 0;
}
