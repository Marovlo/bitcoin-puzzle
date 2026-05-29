#pragma once
#include <cstdint>
#include <cstring>

namespace hash {

// --- SHA-256 for exactly 33 bytes (compressed pubkey) ---
// Single block (64 bytes with padding), no multi-block overhead.

static constexpr uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void sha256_33bytes(const uint8_t pubkey[33], uint8_t hash[32]) {
    uint32_t W[64];
    // Load first 32 bytes as 8 big-endian words
    for (int i = 0; i < 8; ++i) {
        W[i] = ((uint32_t)pubkey[i*4] << 24) | ((uint32_t)pubkey[i*4+1] << 16)
              | ((uint32_t)pubkey[i*4+2] << 8) | (uint32_t)pubkey[i*4+3];
    }
    // Word 8: byte 32 in high byte + 0x80 padding
    W[8] = ((uint32_t)pubkey[32] << 24) | (0x80u << 16);
    // Words 9..13: zeros
    W[9] = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0;
    // Length: 33 * 8 = 264 bits
    W[14] = 0;
    W[15] = 264;

    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(W[i-15], 7) ^ rotr32(W[i-15], 18) ^ (W[i-15] >> 3);
        uint32_t s1 = rotr32(W[i-2], 17) ^ rotr32(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + W[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    a += 0x6a09e667; b += 0xbb67ae85; c += 0x3c6ef372; d += 0xa54ff53a;
    e += 0x510e527f; f += 0x9b05688c; g += 0x1f83d9ab; h += 0x5be0cd19;

    uint32_t H[8] = {a, b, c, d, e, f, g, h};
    for (int i = 0; i < 8; ++i) {
        hash[i*4]   = (uint8_t)(H[i] >> 24);
        hash[i*4+1] = (uint8_t)(H[i] >> 16);
        hash[i*4+2] = (uint8_t)(H[i] >> 8);
        hash[i*4+3] = (uint8_t)(H[i]);
    }
}

// --- RIPEMD-160 for exactly 32 bytes (SHA-256 output) ---

inline uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

static constexpr uint32_t RMD_KL[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
static constexpr uint32_t RMD_KR[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};

static constexpr int RMD_RL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
static constexpr int RMD_RR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
static constexpr int RMD_SL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
static constexpr int RMD_SR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

inline void ripemd160_32bytes(const uint8_t sha_out[32], uint8_t h160[20]) {
    uint32_t X[16];
    for (int i = 0; i < 8; ++i) {
        X[i] = (uint32_t)sha_out[i*4] | ((uint32_t)sha_out[i*4+1] << 8)
             | ((uint32_t)sha_out[i*4+2] << 16) | ((uint32_t)sha_out[i*4+3] << 24);
    }
    X[8] = 0x00000080u;
    for (int i = 9; i < 14; ++i) X[i] = 0;
    X[14] = 256u;
    X[15] = 0;

    uint32_t AL = 0x67452301, BL = 0xEFCDAB89, CL = 0x98BADCFE;
    uint32_t DL = 0x10325476, EL = 0xC3D2E1F0;
    uint32_t AR = AL, BR = BL, CR = CL, DR = DL, ER = EL;

    auto f0 = [](uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; };
    auto f1 = [](uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); };
    auto f2 = [](uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; };
    auto f3 = [](uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); };
    auto f4 = [](uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); };

    using FnType = uint32_t(*)(uint32_t, uint32_t, uint32_t);
    FnType fL[5] = {f0, f1, f2, f3, f4};
    FnType fR[5] = {f4, f3, f2, f1, f0};

    for (int round = 0; round < 5; ++round) {
        for (int j = round * 16; j < (round + 1) * 16; ++j) {
            uint32_t tL = rotl32(AL + fL[round](BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[round], RMD_SL[j]) + EL;
            AL = EL; EL = DL; DL = rotl32(CL, 10); CL = BL; BL = tL;
            uint32_t tR = rotl32(AR + fR[round](BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[round], RMD_SR[j]) + ER;
            AR = ER; ER = DR; DR = rotl32(CR, 10); CR = BR; BR = tR;
        }
    }

    uint32_t H0 = 0x67452301, H1 = 0xEFCDAB89, H2 = 0x98BADCFE;
    uint32_t H3 = 0x10325476, H4 = 0xC3D2E1F0;
    uint32_t t = H1 + CL + DR;
    H1 = H2 + DL + ER;
    H2 = H3 + EL + AR;
    H3 = H4 + AL + BR;
    H4 = H0 + BL + CR;
    H0 = t;

    uint32_t Hf[5] = {H0, H1, H2, H3, H4};
    for (int i = 0; i < 5; ++i) {
        h160[i*4]   = (uint8_t)(Hf[i]);
        h160[i*4+1] = (uint8_t)(Hf[i] >> 8);
        h160[i*4+2] = (uint8_t)(Hf[i] >> 16);
        h160[i*4+3] = (uint8_t)(Hf[i] >> 24);
    }
}

// Combined: privkey -> hash160
inline void pubkey_to_hash160(const uint8_t compressed_pubkey[33], uint8_t h160[20]) {
    uint8_t sha[32];
    sha256_33bytes(compressed_pubkey, sha);
    ripemd160_32bytes(sha, h160);
}

}  // namespace hash
