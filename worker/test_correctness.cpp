// Standalone correctness harness for CPU crypto kernels.
// Computes hash160(compressed pubkey) for each known solved-puzzle private key
// and compares against ground-truth h160 derived from the puzzle address.
//
// Build: see `make test-correct` target. Run from worker/ dir.
#include "kernels/secp256k1.h"
#include "kernels/hash.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

static void hex_to_key(const std::string& hex, uint64_t k[4]) {
    k[0] = k[1] = k[2] = k[3] = 0;
    std::string h = hex;
    if (h.size() > 64) h = h.substr(h.size() - 64);
    // right-aligned big-endian hex -> little-endian limbs
    int n = (int)h.size();
    for (int i = 0; i < n; i++) {
        int nyb = -1;
        char c = h[n - 1 - i];
        if (c >= '0' && c <= '9') nyb = c - '0';
        else if (c >= 'a' && c <= 'f') nyb = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') nyb = c - 'A' + 10;
        else continue;
        int limb = i / 16;
        int shift = (i % 16) * 4;
        k[limb] |= (uint64_t)nyb << shift;
    }
}

// Compute h160 of compressed pubkey for scalar k using the same path as the solver.
static void key_to_h160(const uint64_t k[4], const uint64_t* g_table, uint8_t h160[20]) {
    secp256k1::JacobianPoint P;
    secp256k1::scalar_mul_g_windowed(P, k, g_table);

    // Affinize
    uint64_t zinv[4], zi2[4], zi3[4], ax[4], ay[4];
    secp256k1::mod_inv(zinv, P.Z);
    secp256k1::mod_sqr(zi2, zinv);
    secp256k1::mod_mul(zi3, zi2, zinv);
    secp256k1::mod_mul(ax, P.X, zi2);
    secp256k1::mod_mul(ay, P.Y, zi3);

    uint8_t pubkey[33];
    pubkey[0] = 0x02 | (uint8_t)(ay[0] & 1);
    for (int li = 0; li < 4; li++) {
        uint64_t l = ax[3 - li];
        for (int j = 0; j < 8; j++)
            pubkey[1 + li * 8 + j] = (uint8_t)(l >> (56 - 8 * j));
    }
    hash::pubkey_to_hash160(pubkey, h160);
}

int main() {
    uint64_t* g_table = new uint64_t[secp256k1::G_TABLE_ULONGS];
    secp256k1::build_g_table(g_table);

    std::ifstream f("test_vectors.txt");
    if (!f) { fprintf(stderr, "cannot open test_vectors.txt\n"); return 2; }

    int pass = 0, fail = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string keyhex, expect;
        ss >> keyhex >> expect;
        uint64_t k[4];
        hex_to_key(keyhex, k);
        uint8_t h160[20];
        key_to_h160(k, g_table, h160);
        char got[41];
        for (int i = 0; i < 20; i++) sprintf(got + i * 2, "%02x", h160[i]);
        if (expect == got) {
            pass++;
        } else {
            fail++;
            printf("FAIL key=%s\n  expect %s\n  got    %s\n", keyhex.c_str(), expect.c_str(), got);
        }
    }
    delete[] g_table;
    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail ? 1 : 0;
}
