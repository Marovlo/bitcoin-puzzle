#pragma once
#include <cstdint>
#include <cstring>

#if defined(__SHA__) && defined(__SSE4_1__)
  #define HASH_HAVE_SHANI 1
  #include <immintrin.h>
#endif

#if defined(__ARM_FEATURE_SHA2) || defined(__ARM_FEATURE_CRYPTO)
  #define HASH_HAVE_ARM_SHA2 1
  #include <arm_neon.h>
#endif

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

inline void sha256_33bytes_scalar(const uint8_t pubkey[33], uint8_t hash[32]) {
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

#ifdef HASH_HAVE_SHANI
// SHA-256 of exactly 33 bytes using x86 SHA-NI (single 64-byte block).
// Intel/AMD intrinsic sequence; reference: Jeffrey Walton's public-domain code.
inline void sha256_33bytes_shani(const uint8_t pubkey[33], uint8_t hash[32]) {
    // Build the padded 64-byte message block.
    // 33 message bytes, then 0x80, zero pad, 64-bit big-endian length = 264 bits.
    alignas(16) uint8_t block[64];
    memcpy(block, pubkey, 33);
    block[33] = 0x80;
    memset(block + 34, 0, 64 - 34);
    block[62] = 0x01;  // 264 = 0x108 -> high byte of 16-bit tail
    block[63] = 0x08;

    const __m128i MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);

    // Initial state in memory order {A,B,C,D},{E,F,G,H} (lane0 = first word),
    // matching the canonical loadu convention.
    __m128i state0 = _mm_set_epi32(0xa54ff53a, 0x3c6ef372, 0xbb67ae85, 0x6a09e667); // D C B A
    __m128i state1 = _mm_set_epi32(0x5be0cd19, 0x1f83d9ab, 0x9b05688c, 0x510e527f); // H G F E
    // Shuffle into the order the rnds2 intrinsic expects.
    __m128i tmp = _mm_shuffle_epi32(state0, 0xB1);          // CDAB
    state1 = _mm_shuffle_epi32(state1, 0x1B);               // GHEF -> EFGH lanes
    __m128i abef = _mm_alignr_epi8(tmp, state1, 8);         // ABEF
    __m128i cdgh = _mm_blend_epi16(state1, tmp, 0xF0);      // CDGH
    state0 = abef;
    state1 = cdgh;

    const __m128i ABEF_SAVE = state0;
    const __m128i CDGH_SAVE = state1;

    __m128i msg, msg0, msg1, msg2, msg3, mtmp;

    msg0 = _mm_loadu_si128((const __m128i*)(block + 0));
    msg0 = _mm_shuffle_epi8(msg0, MASK);
    msg1 = _mm_loadu_si128((const __m128i*)(block + 16));
    msg1 = _mm_shuffle_epi8(msg1, MASK);
    msg2 = _mm_loadu_si128((const __m128i*)(block + 32));
    msg2 = _mm_shuffle_epi8(msg2, MASK);
    msg3 = _mm_loadu_si128((const __m128i*)(block + 48));
    msg3 = _mm_shuffle_epi8(msg3, MASK);

    // Rounds 0-3
    msg = _mm_add_epi32(msg0, _mm_set_epi64x(0xE9B5DBA5B5C0FBCFULL, 0x71374491428A2F98ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    // Rounds 4-7
    msg = _mm_add_epi32(msg1, _mm_set_epi64x(0xAB1C5ED5923F82A4ULL, 0x59F111F13956C25BULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);
    // Rounds 8-11
    msg = _mm_add_epi32(msg2, _mm_set_epi64x(0x550C7DC3243185BEULL, 0x12835B01D807AA98ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);
    // Rounds 12-15
    msg = _mm_add_epi32(msg3, _mm_set_epi64x(0xC19BF1749BDC06A7ULL, 0x80DEB1FE72BE5D74ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg3, msg2, 4);
    msg0 = _mm_add_epi32(msg0, mtmp);
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);
    // Rounds 16-19
    msg = _mm_add_epi32(msg0, _mm_set_epi64x(0x240CA1CC0FC19DC6ULL, 0xEFBE4786E49B69C1ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg0, msg3, 4);
    msg1 = _mm_add_epi32(msg1, mtmp);
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);
    // Rounds 20-23
    msg = _mm_add_epi32(msg1, _mm_set_epi64x(0x76F988DA5CB0A9DCULL, 0x4A7484AA2DE92C6FULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg1, msg0, 4);
    msg2 = _mm_add_epi32(msg2, mtmp);
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);
    // Rounds 24-27
    msg = _mm_add_epi32(msg2, _mm_set_epi64x(0xBF597FC7B00327C8ULL, 0xA831C66D983E5152ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg2, msg1, 4);
    msg3 = _mm_add_epi32(msg3, mtmp);
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);
    // Rounds 28-31
    msg = _mm_add_epi32(msg3, _mm_set_epi64x(0x1429296706CA6351ULL, 0xD5A79147C6E00BF3ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg3, msg2, 4);
    msg0 = _mm_add_epi32(msg0, mtmp);
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);
    // Rounds 32-35
    msg = _mm_add_epi32(msg0, _mm_set_epi64x(0x53380D134D2C6DFCULL, 0x2E1B213827B70A85ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg0, msg3, 4);
    msg1 = _mm_add_epi32(msg1, mtmp);
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);
    // Rounds 36-39
    msg = _mm_add_epi32(msg1, _mm_set_epi64x(0x92722C8581C2C92EULL, 0x766A0ABB650A7354ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg1, msg0, 4);
    msg2 = _mm_add_epi32(msg2, mtmp);
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);
    // Rounds 40-43
    msg = _mm_add_epi32(msg2, _mm_set_epi64x(0xC76C51A3C24B8B70ULL, 0xA81A664BA2BFE8A1ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg2, msg1, 4);
    msg3 = _mm_add_epi32(msg3, mtmp);
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);
    // Rounds 44-47
    msg = _mm_add_epi32(msg3, _mm_set_epi64x(0x106AA070F40E3585ULL, 0xD6990624D192E819ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg3, msg2, 4);
    msg0 = _mm_add_epi32(msg0, mtmp);
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);
    // Rounds 48-51
    msg = _mm_add_epi32(msg0, _mm_set_epi64x(0x34B0BCB52748774CULL, 0x1E376C0819A4C116ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg0, msg3, 4);
    msg1 = _mm_add_epi32(msg1, mtmp);
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);
    // Rounds 52-55
    msg = _mm_add_epi32(msg1, _mm_set_epi64x(0x682E6FF35B9CCA4FULL, 0x4ED8AA4A391C0CB3ULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg1, msg0, 4);
    msg2 = _mm_add_epi32(msg2, mtmp);
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    // Rounds 56-59
    msg = _mm_add_epi32(msg2, _mm_set_epi64x(0x8CC7020884C87814ULL, 0x78A5636F748F82EEULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    mtmp = _mm_alignr_epi8(msg2, msg1, 4);
    msg3 = _mm_add_epi32(msg3, mtmp);
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
    // Rounds 60-63
    msg = _mm_add_epi32(msg3, _mm_set_epi64x(0xC67178F2BEF9A3F7ULL, 0xA4506CEB90BEFFFAULL));
    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    msg = _mm_shuffle_epi32(msg, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

    // Add the saved state.
    state0 = _mm_add_epi32(state0, ABEF_SAVE);
    state1 = _mm_add_epi32(state1, CDGH_SAVE);

    // Un-shuffle back to A B C D E F G H.
    tmp    = _mm_shuffle_epi32(state0, 0x1B);    // FEBA
    state1 = _mm_shuffle_epi32(state1, 0xB1);    // DCHG
    state0 = _mm_blend_epi16(tmp, state1, 0xF0); // DCBA
    state1 = _mm_alignr_epi8(state1, tmp, 8);    // ABEF -> HGFE

    // Byte-swap to big-endian and store.
    state0 = _mm_shuffle_epi8(state0, MASK);
    state1 = _mm_shuffle_epi8(state1, MASK);
    _mm_storeu_si128((__m128i*)(hash + 0), state0);
    _mm_storeu_si128((__m128i*)(hash + 16), state1);
}
#endif // HASH_HAVE_SHANI

#ifdef HASH_HAVE_ARM_SHA2
// SHA-256 of exactly 33 bytes using ARMv8 crypto extension (single 64-byte
// block). One message schedule + 64 rounds via vsha256hq/h2q + su0/su1.
inline void sha256_33bytes_arm(const uint8_t pubkey[33], uint8_t hash[32]) {
    // Padded 64-byte block: 33 message bytes, 0x80, zero pad, 64-bit BE length.
    alignas(16) uint8_t block[64];
    memcpy(block, pubkey, 33);
    block[33] = 0x80;
    memset(block + 34, 0, 64 - 34);
    block[62] = 0x01;   // length = 264 bits = 0x108
    block[63] = 0x08;

    // Load 4 message vectors, byte-swap each 32-bit word to big-endian.
    uint32x4_t m0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 0)));
    uint32x4_t m1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 16)));
    uint32x4_t m2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 32)));
    uint32x4_t m3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 48)));

    // Initial state {A,B,C,D}, {E,F,G,H}.
    uint32x4_t s0 = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a};
    uint32x4_t s1 = {0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const uint32x4_t abef0 = s0, cdgh0 = s1;

    uint32x4_t k, t;
    // Helper macro: one group of 4 rounds with state update.
    #define RND4(MSG, KIDX) do {                                  \
        k = vld1q_u32(&SHA256_K[(KIDX)]);                         \
        uint32x4_t w = vaddq_u32((MSG), k);                       \
        uint32x4_t s0o = s0;                                      \
        s0 = vsha256hq_u32(s0, s1, w);                            \
        s1 = vsha256h2q_u32(s1, s0o, w);                          \
    } while (0)

    // schedule + rounds, interleaved (su0/su1 produce next msg vectors)
    RND4(m0, 0);
    uint32x4_t n0 = vsha256su1q_u32(vsha256su0q_u32(m0, m1), m2, m3);
    RND4(m1, 4);
    uint32x4_t n1 = vsha256su1q_u32(vsha256su0q_u32(m1, m2), m3, n0);
    RND4(m2, 8);
    uint32x4_t n2 = vsha256su1q_u32(vsha256su0q_u32(m2, m3), n0, n1);
    RND4(m3, 12);
    uint32x4_t n3 = vsha256su1q_u32(vsha256su0q_u32(m3, n0), n1, n2);
    RND4(n0, 16);
    m0 = vsha256su1q_u32(vsha256su0q_u32(n0, n1), n2, n3);
    RND4(n1, 20);
    m1 = vsha256su1q_u32(vsha256su0q_u32(n1, n2), n3, m0);
    RND4(n2, 24);
    m2 = vsha256su1q_u32(vsha256su0q_u32(n2, n3), m0, m1);
    RND4(n3, 28);
    m3 = vsha256su1q_u32(vsha256su0q_u32(n3, m0), m1, m2);
    RND4(m0, 32);
    n0 = vsha256su1q_u32(vsha256su0q_u32(m0, m1), m2, m3);
    RND4(m1, 36);
    n1 = vsha256su1q_u32(vsha256su0q_u32(m1, m2), m3, n0);
    RND4(m2, 40);
    n2 = vsha256su1q_u32(vsha256su0q_u32(m2, m3), n0, n1);
    RND4(m3, 44);
    n3 = vsha256su1q_u32(vsha256su0q_u32(m3, n0), n1, n2);
    RND4(n0, 48);
    RND4(n1, 52);
    RND4(n2, 56);
    RND4(n3, 60);
    #undef RND4

    s0 = vaddq_u32(s0, abef0);
    s1 = vaddq_u32(s1, cdgh0);

    // Byte-swap back to big-endian and store.
    vst1q_u8(hash + 0,  vrev32q_u8(vreinterpretq_u8_u32(s0)));
    vst1q_u8(hash + 16, vrev32q_u8(vreinterpretq_u8_u32(s1)));
}
#endif // HASH_HAVE_ARM_SHA2

// Dispatch: use SHA-NI / ARMv8-SHA2 when compiled in, else portable scalar.
inline void sha256_33bytes(const uint8_t pubkey[33], uint8_t hash[32]) {
#if defined(HASH_HAVE_SHANI)
    sha256_33bytes_shani(pubkey, hash);
#elif defined(HASH_HAVE_ARM_SHA2)
    sha256_33bytes_arm(pubkey, hash);
#else
    sha256_33bytes_scalar(pubkey, hash);
#endif
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

#if defined(__AVX2__)
  #define HASH_HAVE_AVX2_RMD 1

// --- AVX2 8-way RIPEMD-160 over 8 independent 32-byte inputs ---
// Each 32-byte input is the SHA-256 output; the padded block layout matches
// ripemd160_32bytes (word[8]=0x80, word[14]=256-bit length).
inline __m256i rmd_rotl_v(__m256i x, int n) {
    return _mm256_or_si256(_mm256_slli_epi32(x, n), _mm256_srli_epi32(x, 32 - n));
}
inline __m256i rmd_not(__m256i x) { return _mm256_xor_si256(x, _mm256_set1_epi32(-1)); }
inline __m256i rmd_f0(__m256i x, __m256i y, __m256i z) { return _mm256_xor_si256(_mm256_xor_si256(x, y), z); }
inline __m256i rmd_f1(__m256i x, __m256i y, __m256i z) { return _mm256_or_si256(_mm256_and_si256(x, y), _mm256_andnot_si256(x, z)); }
inline __m256i rmd_f2(__m256i x, __m256i y, __m256i z) { return _mm256_xor_si256(_mm256_or_si256(x, rmd_not(y)), z); }
inline __m256i rmd_f3(__m256i x, __m256i y, __m256i z) { return _mm256_or_si256(_mm256_and_si256(x, z), _mm256_andnot_si256(z, y)); }
inline __m256i rmd_f4(__m256i x, __m256i y, __m256i z) { return _mm256_xor_si256(x, _mm256_or_si256(y, rmd_not(z))); }

// in: 8 contiguous 32-byte SHA outputs; out: 8 contiguous 20-byte digests.
inline void ripemd160_8way(const uint8_t in[8][32], uint8_t out[8][20]) {
    __m256i X[16];
    uint32_t w[8][8];
    for (int m = 0; m < 8; m++)
        for (int j = 0; j < 8; j++)
            w[m][j] = (uint32_t)in[m][j*4] | ((uint32_t)in[m][j*4+1] << 8)
                    | ((uint32_t)in[m][j*4+2] << 16) | ((uint32_t)in[m][j*4+3] << 24);
    for (int j = 0; j < 8; j++)
        X[j] = _mm256_set_epi32(w[7][j], w[6][j], w[5][j], w[4][j], w[3][j], w[2][j], w[1][j], w[0][j]);
    X[8] = _mm256_set1_epi32(0x00000080u);
    for (int j = 9; j < 14; j++) X[j] = _mm256_setzero_si256();
    X[14] = _mm256_set1_epi32(256u);
    X[15] = _mm256_setzero_si256();

    const __m256i I0 = _mm256_set1_epi32(0x67452301), I1 = _mm256_set1_epi32(0xEFCDAB89),
                  I2 = _mm256_set1_epi32(0x98BADCFE), I3 = _mm256_set1_epi32(0x10325476),
                  I4 = _mm256_set1_epi32(0xC3D2E1F0);
    __m256i AL = I0, BL = I1, CL = I2, DL = I3, EL = I4;
    __m256i AR = I0, BR = I1, CR = I2, DR = I3, ER = I4;

    for (int round = 0; round < 5; round++) {
        __m256i KLr = _mm256_set1_epi32((int)RMD_KL[round]);
        __m256i KRr = _mm256_set1_epi32((int)RMD_KR[round]);
        for (int j = round * 16; j < (round + 1) * 16; j++) {
            __m256i fl, fr;
            switch (round) {
                case 0: fl = rmd_f0(BL,CL,DL); fr = rmd_f4(BR,CR,DR); break;
                case 1: fl = rmd_f1(BL,CL,DL); fr = rmd_f3(BR,CR,DR); break;
                case 2: fl = rmd_f2(BL,CL,DL); fr = rmd_f2(BR,CR,DR); break;
                case 3: fl = rmd_f3(BL,CL,DL); fr = rmd_f1(BR,CR,DR); break;
                default: fl = rmd_f4(BL,CL,DL); fr = rmd_f0(BR,CR,DR);
            }
            __m256i tL = _mm256_add_epi32(AL, fl);
            tL = _mm256_add_epi32(tL, X[RMD_RL[j]]); tL = _mm256_add_epi32(tL, KLr);
            tL = _mm256_add_epi32(rmd_rotl_v(tL, RMD_SL[j]), EL);
            AL = EL; EL = DL; DL = rmd_rotl_v(CL, 10); CL = BL; BL = tL;
            __m256i tR = _mm256_add_epi32(AR, fr);
            tR = _mm256_add_epi32(tR, X[RMD_RR[j]]); tR = _mm256_add_epi32(tR, KRr);
            tR = _mm256_add_epi32(rmd_rotl_v(tR, RMD_SR[j]), ER);
            AR = ER; ER = DR; DR = rmd_rotl_v(CR, 10); CR = BR; BR = tR;
        }
    }

    __m256i H0 = _mm256_add_epi32(_mm256_add_epi32(I1, CL), DR);
    __m256i H1 = _mm256_add_epi32(_mm256_add_epi32(I2, DL), ER);
    __m256i H2 = _mm256_add_epi32(_mm256_add_epi32(I3, EL), AR);
    __m256i H3 = _mm256_add_epi32(_mm256_add_epi32(I4, AL), BR);
    __m256i H4 = _mm256_add_epi32(_mm256_add_epi32(I0, BL), CR);
    uint32_t h0[8], h1[8], h2[8], h3[8], h4[8];
    _mm256_storeu_si256((__m256i*)h0, H0); _mm256_storeu_si256((__m256i*)h1, H1);
    _mm256_storeu_si256((__m256i*)h2, H2); _mm256_storeu_si256((__m256i*)h3, H3);
    _mm256_storeu_si256((__m256i*)h4, H4);
    for (int m = 0; m < 8; m++) {
        uint32_t H[5] = {h0[m], h1[m], h2[m], h3[m], h4[m]};
        for (int i = 0; i < 5; i++) {
            out[m][i*4]   = (uint8_t)(H[i]);
            out[m][i*4+1] = (uint8_t)(H[i] >> 8);
            out[m][i*4+2] = (uint8_t)(H[i] >> 16);
            out[m][i*4+3] = (uint8_t)(H[i] >> 24);
        }
    }
}

// 8 compressed pubkeys -> 8 hash160 (SHA-256 each, then one 8-way RIPEMD-160).
inline void pubkey_to_hash160_8way(const uint8_t pubkeys[8][33], uint8_t h160[8][20]) {
    uint8_t sha[8][32];
    for (int m = 0; m < 8; m++) sha256_33bytes(pubkeys[m], sha[m]);
    ripemd160_8way(sha, h160);
}
#endif // HASH_HAVE_AVX2_RMD

}  // namespace hash
