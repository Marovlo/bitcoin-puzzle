/**
 * Bitcoin Puzzle OpenCL kernel.
 *
 * Portable OpenCL C port of the CUDA/HIP kernel. Targets AMD RDNA2 (RX 6800 XT,
 * gfx1030) via the stock Adrenalin driver, but stays generic: it only relies on
 * core OpenCL 1.2 integer features (32x32->64 mul, byte-addressable stores).
 *
 * Why OpenCL on Windows/AMD:
 *   - HIP for Windows officially supports RDNA3+ only (gfx1030 is marked
 *     unsupported by AMD), so the HIP backend is a dead end on this GPU.
 *   - OpenCL ships with the normal graphics driver: no SDK install, and the
 *     kernel is compiled at runtime by the driver, so only a host C++ compiler
 *     is needed to build.
 *
 * Arithmetic notes for RDNA2:
 *   - No native 64x64->128 multiply. 256-bit field mul is built from 32-bit
 *     mul_hi/mul_lo pairs, which map 1:1 onto v_mul_hi_u32 / v_mul_lo_u32.
 *   - Wavefront size is 32 (same as an NVIDIA warp), so per-thread control flow
 *     divergence costs a full 32-lane replay just like on NVIDIA.
 */

// ALGORITHM
//   For each work-item (one candidate private key k):
//     1. P = k * G   (windowed scalar mul over a precomputed 32x256 affine table)
//     2. affinize P  (one modular inversion)
//     3. compressed pubkey -> SHA-256 -> RIPEMD-160
//     4. compare against the target hash160
//
// This is the "one scalar mul per key" baseline that CUDA/HIP also use. It is
// kept deliberately simple so the backend can be validated end to end before
// algorithmic tuning.

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// ==========================================================================
// 256-bit field arithmetic mod p = 2^256 - 2^32 - 977
// ==========================================================================

// p limbs, little-endian
#define P0 0xFFFFFFFEFFFFFC2FUL
#define P1 0xFFFFFFFFFFFFFFFFUL
#define P2 0xFFFFFFFFFFFFFFFFUL
#define P3 0xFFFFFFFFFFFFFFFFUL

// 2^256 mod p
#define K0 0x1000003D1UL

// High 64 bits of a*b. OpenCL exposes mul_hi() for 32-bit ints only, so the
// 64-bit case is spelled out: the four 32x32 partial products give the compiler
// the exact structure it can map to v_mul_lo_u32 / v_mul_hi_u32.
static inline ulong mul_hi64(ulong a, ulong b) {
    const uint a0 = (uint)a, a1 = (uint)(a >> 32);
    const uint b0 = (uint)b, b1 = (uint)(b >> 32);

    const ulong p00 = (ulong)a0 * b0;
    const ulong p01 = (ulong)a0 * b1;
    const ulong p10 = (ulong)a1 * b0;
    const ulong p11 = (ulong)a1 * b1;

    const ulong mid = p01 + p10;
    const ulong mid_carry = (mid < p01) ? 1UL : 0UL;   // mid overflowed 2^64
    const ulong low = p00 + (mid << 32);
    const ulong low_carry = (low < p00) ? 1UL : 0UL;

    return p11 + (mid >> 32) + (mid_carry << 32) + low_carry;
}

static inline void fcopy(ulong d[4], const ulong s[4]) {
    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
}

static inline void fzero(ulong d[4]) { d[0] = 0; d[1] = 0; d[2] = 0; d[3] = 0; }

static inline bool fis_zero(const ulong a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0UL;
}

static inline ulong add_carry64(ulong a, ulong b, ulong *carry) {
    const ulong sum = a + b + *carry;
    *carry = (sum < a || (*carry && sum == a)) ? 1UL : 0UL;
    return sum;
}

static inline ulong sub_borrow64(ulong a, ulong b, ulong *borrow) {
    const ulong diff = a - b - *borrow;
    *borrow = (a < b + *borrow || (*borrow && b == 0xFFFFFFFFFFFFFFFFUL)) ? 1UL : 0UL;
    return diff;
}

static inline bool ge_p(const ulong a[4]) {
    if (a[3] != P3) return a[3] > P3;
    if (a[2] != P2) return a[2] > P2;
    if (a[1] != P1) return a[1] > P1;
    return a[0] >= P0;
}

static inline void mod_add(ulong r[4], const ulong a[4], const ulong b[4]) {
    ulong c = 0;
    r[0] = add_carry64(a[0], b[0], &c);
    r[1] = add_carry64(a[1], b[1], &c);
    r[2] = add_carry64(a[2], b[2], &c);
    r[3] = add_carry64(a[3], b[3], &c);
    if (c) {
        ulong cc = 0;
        r[0] = add_carry64(r[0], K0, &cc);
        r[1] = add_carry64(r[1], 0, &cc);
        r[2] = add_carry64(r[2], 0, &cc);
        r[3] = add_carry64(r[3], 0, &cc);
    }
    if (ge_p(r)) {
        ulong bo = 0;
        r[0] = sub_borrow64(r[0], P0, &bo);
        r[1] = sub_borrow64(r[1], P1, &bo);
        r[2] = sub_borrow64(r[2], P2, &bo);
        r[3] = sub_borrow64(r[3], P3, &bo);
    }
}

static inline void mod_sub(ulong r[4], const ulong a[4], const ulong b[4]) {
    ulong bo = 0;
    r[0] = sub_borrow64(a[0], b[0], &bo);
    r[1] = sub_borrow64(a[1], b[1], &bo);
    r[2] = sub_borrow64(a[2], b[2], &bo);
    r[3] = sub_borrow64(a[3], b[3], &bo);
    if (bo) {
        ulong c = 0;
        r[0] = add_carry64(r[0], P0, &c);
        r[1] = add_carry64(r[1], P1, &c);
        r[2] = add_carry64(r[2], P2, &c);
        r[3] = add_carry64(r[3], P3, &c);
    }
}

// 256x256 -> 512, then fold the high half back in via 2^256 = K0 (mod p).
static inline void mod_mul(ulong r[4], const ulong a[4], const ulong b[4]) {
    ulong w[8];
    for (int i = 0; i < 8; i++) w[i] = 0;

    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            const ulong lo = a[i] * b[j];
            const ulong hi = mul_hi64(a[i], b[j]);
            ulong s = w[i + j] + lo;
            ulong c1 = (s < w[i + j]) ? 1UL : 0UL;
            s += carry;
            c1 += (s < carry) ? 1UL : 0UL;
            w[i + j] = s;
            carry = hi + c1;
        }
        w[i + 4] = carry;
    }

    // h = w[4..7] * K0  (5 limbs)
    ulong h[5];
    for (int i = 0; i < 5; i++) h[i] = 0;
    {
        ulong carry = 0;
        for (int i = 0; i < 4; i++) {
            const ulong lo = w[4 + i] * K0;
            const ulong hi = mul_hi64(w[4 + i], K0);
            ulong s = h[i] + lo;
            ulong c1 = (s < h[i]) ? 1UL : 0UL;
            s += carry;
            c1 += (s < carry) ? 1UL : 0UL;
            h[i] = s;
            carry = hi + c1;
        }
        h[4] = carry;
    }

    ulong c = 0;
    ulong t0 = add_carry64(w[0], h[0], &c);
    ulong t1 = add_carry64(w[1], h[1], &c);
    ulong t2 = add_carry64(w[2], h[2], &c);
    ulong t3 = add_carry64(w[3], h[3], &c);
    ulong overflow = c + h[4];

    if (overflow) {
        const ulong ol = overflow * K0;
        const ulong oh = mul_hi64(overflow, K0);
        ulong c2 = 0;
        t0 = add_carry64(t0, ol, &c2);
        t1 = add_carry64(t1, oh, &c2);
        t2 = add_carry64(t2, 0, &c2);
        t3 = add_carry64(t3, 0, &c2);
        if (c2) {
            ulong c3 = 0;
            t0 = add_carry64(t0, K0, &c3);
            t1 = add_carry64(t1, 0, &c3);
            t2 = add_carry64(t2, 0, &c3);
            t3 = add_carry64(t3, 0, &c3);
        }
    }

    ulong t[4] = { t0, t1, t2, t3 };
    if (ge_p(t)) {
        ulong bo = 0;
        t[0] = sub_borrow64(t[0], P0, &bo);
        t[1] = sub_borrow64(t[1], P1, &bo);
        t[2] = sub_borrow64(t[2], P2, &bo);
        t[3] = sub_borrow64(t[3], P3, &bo);
    }
    fcopy(r, t);
}

static inline void mod_sqr(ulong r[4], const ulong a[4]) { mod_mul(r, a, a); }

// a^(p-2) mod p via the libsecp256k1 addition chain.
static inline void mod_inv(ulong r[4], const ulong a[4]) {
    ulong x2[4], x3[4], x6[4], x9[4], x11[4], x22[4], x44[4];
    ulong x88[4], x176[4], x220[4], x223[4], t[4];

    mod_sqr(x2, a); mod_mul(x2, x2, a);
    mod_sqr(x3, x2); mod_mul(x3, x3, a);
    mod_sqr(x6, x3); mod_sqr(x6, x6); mod_sqr(x6, x6); mod_mul(x6, x6, x3);
    mod_sqr(x9, x6); mod_sqr(x9, x9); mod_sqr(x9, x9); mod_mul(x9, x9, x3);
    mod_sqr(x11, x9); mod_sqr(x11, x11); mod_mul(x11, x11, x2);

    fcopy(x22, x11);
    for (int i = 0; i < 11; i++) mod_sqr(x22, x22);
    mod_mul(x22, x22, x11);

    fcopy(x44, x22);
    for (int i = 0; i < 22; i++) mod_sqr(x44, x44);
    mod_mul(x44, x44, x22);

    fcopy(x88, x44);
    for (int i = 0; i < 44; i++) mod_sqr(x88, x88);
    mod_mul(x88, x88, x44);

    fcopy(x176, x88);
    for (int i = 0; i < 88; i++) mod_sqr(x176, x176);
    mod_mul(x176, x176, x88);

    fcopy(x220, x176);
    for (int i = 0; i < 44; i++) mod_sqr(x220, x220);
    mod_mul(x220, x220, x44);

    mod_sqr(x223, x220); mod_sqr(x223, x223); mod_sqr(x223, x223);
    mod_mul(x223, x223, x3);

    fcopy(t, x223);
    for (int i = 0; i < 23; i++) mod_sqr(t, t);
    mod_mul(t, t, x22);
    for (int i = 0; i < 5; i++) mod_sqr(t, t);
    mod_mul(t, t, a);
    for (int i = 0; i < 3; i++) mod_sqr(t, t);
    mod_mul(t, t, x2);
    mod_sqr(t, t); mod_sqr(t, t);
    mod_mul(t, t, a);

    fcopy(r, t);
}

// ==========================================================================
// Elliptic curve: Jacobian coordinates
// ==========================================================================

// R = 2*P
static inline void jac_double(ulong rx[4], ulong ry[4], ulong rz[4],
                              const ulong px[4], const ulong py[4], const ulong pz[4]) {
    if (fis_zero(pz) || fis_zero(py)) { fzero(rx); fzero(ry); fzero(rz); return; }

    ulong y2[4], s[4], m[4], t[4], y4[4];
    mod_sqr(y2, py);
    mod_mul(s, px, y2);
    mod_add(s, s, s); mod_add(s, s, s);          // S = 4*X*Y^2
    mod_sqr(m, px);
    mod_add(t, m, m); mod_add(m, t, m);          // M = 3*X^2  (a = 0)

    mod_sqr(rx, m);
    mod_sub(rx, rx, s); mod_sub(rx, rx, s);      // X' = M^2 - 2S
    mod_sub(t, s, rx); mod_mul(t, m, t);
    mod_sqr(y4, y2);
    mod_add(y4, y4, y4); mod_add(y4, y4, y4); mod_add(y4, y4, y4);
    mod_sub(ry, t, y4);                          // Y' = M(S - X') - 8Y^4
    mod_mul(rz, py, pz); mod_add(rz, rz, rz);    // Z' = 2*Y*Z
}

// R = P (Jacobian) + Q (affine)
static inline void jac_add_mixed(ulong rx[4], ulong ry[4], ulong rz[4],
                                 const ulong px[4], const ulong py[4], const ulong pz[4],
                                 const ulong qx[4], const ulong qy[4]) {
    if (fis_zero(qx) && fis_zero(qy)) {
        fcopy(rx, px); fcopy(ry, py); fcopy(rz, pz); return;
    }
    if (fis_zero(pz)) {
        fcopy(rx, qx); fcopy(ry, qy);
        rz[0] = 1; rz[1] = 0; rz[2] = 0; rz[3] = 0;
        return;
    }

    ulong z2[4], z3[4], u2[4], s2[4], h[4], rr[4], h2[4], h3[4], v[4], tmp[4];
    mod_sqr(z2, pz);
    mod_mul(z3, z2, pz);
    mod_mul(u2, qx, z2);
    mod_mul(s2, qy, z3);
    mod_sub(h, u2, px);
    mod_sub(rr, s2, py);

    if (fis_zero(h)) {
        if (fis_zero(rr)) {
            jac_double(rx, ry, rz, px, py, pz);
        } else {
            fzero(rx); fzero(ry); fzero(rz);
        }
        return;
    }

    mod_sqr(h2, h);
    mod_mul(h3, h2, h);
    mod_mul(v, px, h2);
    mod_sqr(tmp, rr);
    mod_sub(tmp, tmp, h3);
    mod_sub(tmp, tmp, v);
    mod_sub(rx, tmp, v);            // X' = r^2 - H^3 - 2V

    mod_sub(tmp, v, rx);
    mod_mul(tmp, rr, tmp);
    ulong yh3[4];
    mod_mul(yh3, py, h3);
    mod_sub(ry, tmp, yh3);          // Y' = r(V - X') - Y*H^3
    mod_mul(rz, pz, h);             // Z' = Z*H
}

// Windowed scalar multiplication against the precomputed table.
// Table layout: 32 windows x 256 entries, entry d = affine(d * 2^(8*w) * G),
// each entry being 8 ulongs: X[4] then Y[4].
static inline void scalar_mul_g(ulong rx[4], ulong ry[4], ulong rz[4],
                                const ulong scalar[4],
                                __global const ulong *g_table) {
    fzero(rx); fzero(ry); fzero(rz);

    #pragma unroll 1
    for (uint w = 0; w < 32; w++) {
        const uint byte_val = (uint)((scalar[w >> 3] >> (8 * (w & 7))) & 0xFFUL);
        if (byte_val == 0) continue;

        __global const ulong *entry = g_table + (size_t)(w * 256 + byte_val) * 8;

        // Pull the affine entry into private registers first: jac_add_mixed works
        // entirely in the private address space, and the explicit load also gives
        // the compiler a clean prefetch point.
        ulong qx[4], qy[4];
        qx[0] = entry[0]; qx[1] = entry[1]; qx[2] = entry[2]; qx[3] = entry[3];
        qy[0] = entry[4]; qy[1] = entry[5]; qy[2] = entry[6]; qy[3] = entry[7];

        ulong nx[4], ny[4], nz[4];
        jac_add_mixed(nx, ny, nz, rx, ry, rz, qx, qy);
        fcopy(rx, nx); fcopy(ry, ny); fcopy(rz, nz);
    }
}

// ==========================================================================
// Hashing
// ==========================================================================

__constant uint SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static inline uint rotr32(uint x, int n) { return (x >> n) | (x << (32 - n)); }
static inline uint rotl32(uint x, int n) { return (x << n) | (x >> (32 - n)); }

// SHA-256 over exactly 33 bytes (a compressed secp256k1 pubkey): one padded
// 64-byte block, so the whole compression runs on a single 16-word schedule
// held in registers.
static inline void sha256_33(const uchar pk[33], uchar hash[32]) {
    uint W[16];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        W[i] = ((uint)pk[i * 4] << 24) | ((uint)pk[i * 4 + 1] << 16) |
               ((uint)pk[i * 4 + 2] << 8) | (uint)pk[i * 4 + 3];
    }
    // byte 32, then 0x80 pad, zero fill, 33*8 = 264-bit length in the last word
    W[8] = ((uint)pk[32] << 24) | (0x80u << 16);
    W[9] = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0; W[14] = 0;
    W[15] = 264u;

    uint a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au;
    uint e = 0x510e527fu, f = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint w;
        if (i < 16) {
            w = W[i];
        } else {
            const uint w15 = W[(i + 1) & 15];
            const uint w2  = W[(i + 14) & 15];
            const uint s0 = rotr32(w15, 7) ^ rotr32(w15, 18) ^ (w15 >> 3);
            const uint s1 = rotr32(w2, 17) ^ rotr32(w2, 19) ^ (w2 >> 10);
            w = W[i & 15] + s0 + W[(i + 9) & 15] + s1;
            W[i & 15] = w;
        }

        const uint S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint ch = (e & f) ^ (~e & g);
        const uint t1 = h + S1 + ch + SHA256_K[i] + w;
        const uint S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint maj = (a & b) ^ (a & c) ^ (b & c);
        const uint t2 = S0 + maj;

        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    const uint H[8] = {
        a + 0x6a09e667u, b + 0xbb67ae85u, c + 0x3c6ef372u, d + 0xa54ff53au,
        e + 0x510e527fu, f + 0x9b05688cu, g + 0x1f83d9abu, h + 0x5be0cd19u
    };
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i * 4]     = (uchar)(H[i] >> 24);
        hash[i * 4 + 1] = (uchar)(H[i] >> 16);
        hash[i * 4 + 2] = (uchar)(H[i] >> 8);
        hash[i * 4 + 3] = (uchar)(H[i]);
    }
}

__constant uint RKL[5] = { 0x00000000u, 0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xA953FD4Eu };
__constant uint RKR[5] = { 0x50A28BE6u, 0x5C4DD124u, 0x6D703EF3u, 0x7A6D76E9u, 0x00000000u };

__constant uchar RRL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
__constant uchar RRR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
__constant uchar RSL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
__constant uchar RSR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

// RIPEMD-160 over exactly 32 bytes (a SHA-256 digest): one padded block where
// word[8] = 0x80 and word[14] = 256 (bits). Fully unrolled so the message array
// stays in registers despite the data-dependent word indices.
static inline void ripemd160_32(const uchar sha[32], uchar h160[20]) {
    uint X[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) X[i] = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        X[i] = (uint)sha[i * 4] | ((uint)sha[i * 4 + 1] << 8) |
               ((uint)sha[i * 4 + 2] << 16) | ((uint)sha[i * 4 + 3] << 24);
    }
    X[8] = 0x00000080u;
    X[14] = 256u;

    uint AL = 0x67452301u, BL = 0xEFCDAB89u, CL = 0x98BADCFEu;
    uint DL = 0x10325476u, EL = 0xC3D2E1F0u;
    uint AR = AL, BR = BL, CR = CL, DR = DL, ER = EL;

    #pragma unroll
    for (int j = 0; j < 80; j++) {
        uint fl, fr;
        const int round = j >> 4;
        if (round == 0) {
            fl = BL ^ CL ^ DL;
            fr = BR ^ (CR | ~DR);
        } else if (round == 1) {
            fl = (BL & CL) | (~BL & DL);
            fr = (BR & DR) | (CR & ~DR);
        } else if (round == 2) {
            fl = (BL | ~CL) ^ DL;
            fr = (BR | ~CR) ^ DR;
        } else if (round == 3) {
            fl = (BL & DL) | (CL & ~DL);
            fr = (BR & CR) | (~BR & DR);
        } else {
            fl = BL ^ (CL | ~DL);
            fr = BR ^ CR ^ DR;
        }

        const uint tL = rotl32(AL + fl + X[RRL[j]] + RKL[round], RSL[j]) + EL;
        AL = EL; EL = DL; DL = rotl32(CL, 10); CL = BL; BL = tL;

        const uint tR = rotl32(AR + fr + X[RRR[j]] + RKR[round], RSR[j]) + ER;
        AR = ER; ER = DR; DR = rotl32(CR, 10); CR = BR; BR = tR;
    }

    // Final combine. The right lane's C is paired with the left lane's D and so
    // on; note the pairing is offset by one against the initial value order.
    const uint H0 = 0xEFCDAB89u + CL + DR;
    const uint H1 = 0x98BADCFEu + DL + ER;
    const uint H2 = 0x10325476u + EL + AR;
    const uint H3 = 0xC3D2E1F0u + AL + BR;
    const uint H4 = 0x67452301u + BL + CR;

    const uint Hf[5] = { H0, H1, H2, H3, H4 };
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        h160[i * 4]     = (uchar)(Hf[i]);
        h160[i * 4 + 1] = (uchar)(Hf[i] >> 8);
        h160[i * 4 + 2] = (uchar)(Hf[i] >> 16);
        h160[i * 4 + 3] = (uchar)(Hf[i] >> 24);
    }
}

// ==========================================================================
// Entry point
// ==========================================================================

// Baseline kernel: one work-item per candidate key.
//
// start_lo/start_hi form the 128-bit start index; the per-item scalar is
// (start + gid) mod 2^128, which is far below the group order so no reduction
// is needed. g_table holds the precomputed multiples of G.
//
// Kept as a reference implementation and as a fallback; puzzle_search_group is
// the one actually used (see the note there).
__kernel void puzzle_search_naive(
    __global const ulong *g_table,
    __global const uint  *target_h160,   // 5 words, little-endian
    const ulong start_lo,
    const ulong start_hi,
    const ulong total_keys,
    __global ulong *match_lo,
    __global ulong *match_hi,
    __global uint  *match_found)
{
    const ulong gid = (ulong)get_global_id(0);
    if (gid >= total_keys) return;
    if (*match_found != 0u) return;

    // k = start + gid
    ulong k[4];
    const ulong sum = start_lo + gid;
    k[0] = sum;
    k[1] = start_hi + ((sum < start_lo) ? 1UL : 0UL);
    k[2] = 0; k[3] = 0;

    ulong px[4], py[4], pz[4];
    scalar_mul_g(px, py, pz, k, g_table);
    if (fis_zero(pz)) return;

    // Affine conversion
    ulong zi[4], zi2[4], zi3[4], ax[4], ay[4];
    mod_inv(zi, pz);
    mod_sqr(zi2, zi);
    mod_mul(zi3, zi2, zi);
    mod_mul(ax, px, zi2);
    mod_mul(ay, py, zi3);

    // Compressed pubkey: 0x02/0x03 || X (big-endian)
    uchar pubkey[33];
    pubkey[0] = (uchar)(0x02 | (uint)(ay[0] & 1UL));
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const ulong limb = ax[3 - i];
        #pragma unroll
        for (int j = 0; j < 8; j++)
            pubkey[1 + i * 8 + j] = (uchar)(limb >> (56 - 8 * j));
    }

    uchar sha[32];
    uchar h160[20];
    sha256_33(pubkey, sha);
    ripemd160_32(sha, h160);

    // Compare as five little-endian words (matches the host byte order).
    bool match = true;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        const uint got = (uint)h160[i * 4] | ((uint)h160[i * 4 + 1] << 8) |
                         ((uint)h160[i * 4 + 2] << 16) | ((uint)h160[i * 4 + 3] << 24);
        if (got != target_h160[i]) { match = false; break; }
    }

    if (match) {
        match_lo[0] = k[0];
        match_hi[0] = k[1];
        *match_found = 1u;
    }
}

// ==========================================================================
// Group search kernel (the one that is actually used)
// ==========================================================================
//
// The naive kernel above costs one windowed scalar mul (~32 mixed additions,
// ~350 field muls) *plus* one modular inversion (~255 squarings) per candidate
// key. That is ~600 field muls per key, and it is why the naive GPU kernel is
// only on par with the CPU backend.
//
// This kernel amortises both costs, using the same construction as backend_cpu.h:
//
//   * A "group" is GROUP_SIZE = 2*GROUP_H+1 consecutive keys centred on C.
//     The multiplicand table gives i*G affinely for i = 1..GROUP_H, and
//     C+i*G / C-i*G share the x-difference (i*G.x - C.x), so ONE inverse serves
//     both points of each pair.
//   * One Montgomery batch inversion covers the whole group, replacing
//     GROUP_H+1 separate inversions with a single one plus ~3 field muls each.
//   * The centre advances by C += GROUP_SIZE*G between groups, which needs a
//     single field mul plus the batch inverse that is already computed.
//
// Net cost drops from ~600 field muls/key to roughly 11-20 (see GROUP_H below),
// which is what turns the GPU from "no faster than the CPU" into the fastest
// backend on the machine.

// Half-width of a group. A larger H spreads the single modular inversion over
// more keys, but the batch-inversion prefix pre[H+1] lives in registers, so it
// trades against occupancy. Measured on RX 6800 XT (gfx1030), 50M-key batches,
// across the values swept:
//     H = 8  -> 476 MK/s
//     H = 16 -> 623 MK/s
//     H = 32 -> 652 MK/s  (chosen)
#define GROUP_H 32
#define GROUP_SIZE (2 * GROUP_H + 1)

// Groups handled per work-item. The first group pays a full scalar mul for its
// centre; later groups ride the ~3-mul centre advance. Measured: 8 beat 16
// (652 vs 567 MK/s) — longer kernels lose more to tail imbalance than they gain
// from amortising that first scalar mul.
#define GROUPS_PER_ITEM 8

// Exposes the geometry above so the host launch code cannot drift out of sync.
__kernel void kernel_params(__global uint *out) {
    out[0] = (uint)GROUP_H;
    out[1] = (uint)GROUP_SIZE;
    out[2] = (uint)GROUPS_PER_ITEM;
}

// Hash the candidate at `idx` and compare. On a match the key is published and
// the caller unwinds; at most one key in any batch can match.
static inline bool point_matches(const ulong ax[4], const ulong ay[4],
                                 const ulong idx, const ulong total_keys,
                                 const ulong start_lo, const ulong start_hi,
                                 __global const uint *target_h160,
                                 __global ulong *match_lo, __global ulong *match_hi,
                                 __global uint *match_found) {
    if (idx >= total_keys) return false;

    uchar pubkey[33];
    pubkey[0] = (uchar)(0x02 | (uint)(ay[0] & 1UL));
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const ulong limb = ax[3 - i];
        #pragma unroll
        for (int j = 0; j < 8; j++)
            pubkey[1 + i * 8 + j] = (uchar)(limb >> (56 - 8 * j));
    }

    uchar sha[32];
    uchar h160[20];
    sha256_33(pubkey, sha);
    ripemd160_32(sha, h160);

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        const uint got = (uint)h160[i * 4] | ((uint)h160[i * 4 + 1] << 8) |
                         ((uint)h160[i * 4 + 2] << 16) | ((uint)h160[i * 4 + 3] << 24);
        if (got != target_h160[i]) return false;
    }

    const ulong klo = start_lo + idx;
    match_lo[0] = klo;
    match_hi[0] = start_hi + ((klo < start_lo) ? 1UL : 0UL);
    *match_found = 1u;
    return true;
}

// (x3,y3) = (cx,cy) + (qx,qy) in affine coordinates.
// inv_den must be 1/(qx - cx); the caller shares one inv_den between ±i*G.
static inline void affine_add(const ulong cx[4], const ulong cy[4],
                              const ulong qx[4], const ulong qy[4],
                              const ulong inv_den[4],
                              ulong x3[4], ulong y3[4]) {
    ulong dy[4], s[4], t[4];
    mod_sub(dy, qy, cy);
    mod_mul(s, dy, inv_den);          // slope
    mod_sqr(x3, s);
    mod_sub(x3, x3, cx);
    mod_sub(x3, x3, qx);              // x3 = s^2 - cx - qx
    mod_sub(t, cx, x3);
    mod_mul(y3, s, t);
    mod_sub(y3, y3, cy);              // y3 = s(cx - x3) - cy
}

// Safety net for the (astronomically rare) group whose centre x collides with a
// table x, which would make a denominator zero. Recomputes the group's keys the
// slow way so no candidate is ever silently skipped.
static inline void group_fallback(const ulong group_base, const ulong total_keys,
                                  const ulong start_lo, const ulong start_hi,
                                  __global const ulong *g_table,
                                  __global const uint *target_h160,
                                  __global ulong *match_lo, __global ulong *match_hi,
                                  __global uint *match_found) {
    #pragma unroll 1
    for (int m = 0; m < GROUP_SIZE; m++) {
        const ulong idx = group_base + (ulong)m;
        if (idx >= total_keys) return;

        ulong k[4];
        const ulong lo = start_lo + idx;
        k[0] = lo;
        k[1] = start_hi + ((lo < start_lo) ? 1UL : 0UL);
        k[2] = 0; k[3] = 0;

        ulong px[4], py[4], pz[4];
        scalar_mul_g(px, py, pz, k, g_table);
        if (fis_zero(pz)) continue;

        ulong zi[4], zi2[4], zi3[4], ax[4], ay[4];
        mod_inv(zi, pz);
        mod_sqr(zi2, zi);
        mod_mul(zi3, zi2, zi);
        mod_mul(ax, px, zi2);
        mod_mul(ay, py, zi3);

        if (point_matches(ax, ay, idx, total_keys, start_lo, start_hi,
                          target_h160, match_lo, match_hi, match_found))
            return;
    }
}

// grp_table layout (8 ulongs per entry, affine):
//   entry 0        : GROUP_SIZE * G
//   entry i, 1..H  : i * G
__kernel void puzzle_search_group(
    __global const ulong *g_table,
    __global const ulong *grp_table,
    __global const uint  *target_h160,   // 5 words, little-endian
    const ulong start_lo,
    const ulong start_hi,
    const ulong total_keys,
    __global ulong *match_lo,
    __global ulong *match_hi,
    __global uint  *match_found)
{
    const ulong item = (ulong)get_global_id(0);
    const ulong total_groups = (total_keys + GROUP_SIZE - 1) / GROUP_SIZE;
    const ulong first_group = item * GROUPS_PER_ITEM;
    if (first_group >= total_groups) return;
    if (*match_found != 0u) return;

    ulong last_group = first_group + GROUPS_PER_ITEM;
    if (last_group > total_groups) last_group = total_groups;

    ulong step_x[4], step_y[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        step_x[i] = grp_table[i];
        step_y[i] = grp_table[4 + i];
    }

    // p, for negating i*G.y (no point of order 2 exists on secp256k1, so the
    // y values are never zero and p - y is always the true negation).
    ulong PM[4];
    PM[0] = P0; PM[1] = P1; PM[2] = P2; PM[3] = P3;

    ulong Cx[4], Cy[4];
    bool centre_valid = false;

    for (ulong g = first_group; g < last_group; ++g) {
        if (*match_found != 0u) return;
        const ulong group_base = g * GROUP_SIZE;

        if (!centre_valid) {
            // One windowed scalar mul sets up the centre for this group.
            const ulong centre_index = group_base + GROUP_H;
            ulong k[4];
            const ulong lo = start_lo + centre_index;
            k[0] = lo;
            k[1] = start_hi + ((lo < start_lo) ? 1UL : 0UL);
            k[2] = 0; k[3] = 0;

            ulong px[4], py[4], pz[4];
            scalar_mul_g(px, py, pz, k, g_table);
            if (fis_zero(pz)) {
                // Centre at infinity: cannot happen for a real key range, but
                // fall back rather than emit nothing.
                group_fallback(group_base, total_keys, start_lo, start_hi, g_table,
                               target_h160, match_lo, match_hi, match_found);
                continue;
            }

            ulong zi[4], zi2[4], zi3[4];
            mod_inv(zi, pz);
            mod_sqr(zi2, zi);
            mod_mul(zi3, zi2, zi);
            mod_mul(Cx, px, zi2);
            mod_mul(Cy, py, zi3);
            centre_valid = true;
        }

        // ---- denominators and the Montgomery product prefix ----
        // den[i] = i*G.x - Cx for i = 1..H, and den[0] for the centre advance.
        // Only the prefix is kept; the denominators are cheap to recompute.
        ulong pre[GROUP_H + 1][4];
        {
            ulong den[4];
            mod_sub(den, step_x, Cx);
            if (fis_zero(den)) {
                group_fallback(group_base, total_keys, start_lo, start_hi, g_table,
                               target_h160, match_lo, match_hi, match_found);
                centre_valid = false;
                continue;
            }
            fcopy(pre[0], den);
        }

        bool degenerate = false;
        #pragma unroll
        for (int i = 1; i <= GROUP_H; i++) {
            ulong igx[4], den[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) igx[j] = grp_table[(size_t)i * 8 + j];
            mod_sub(den, igx, Cx);
            if (fis_zero(den)) degenerate = true;   // note: pre[i] is then bogus
            mod_mul(pre[i], pre[i - 1], den);
        }

        if (degenerate) {
            group_fallback(group_base, total_keys, start_lo, start_hi, g_table,
                           target_h160, match_lo, match_hi, match_found);
            centre_valid = false;
            continue;
        }

        // ---- one inverse for the entire group ----
        ulong acc[4];
        mod_inv(acc, pre[GROUP_H]);

        // Walking backwards recovers each 1/den[i] from the running product.
        // The ± pair is emitted immediately, so no inverse array is materialised
        // (that is what keeps the register footprint low enough to stay on-chip).
        #pragma unroll
        for (int k = 0; k < GROUP_H; k++) {
            const int i = GROUP_H - k;

            ulong igx[4], igy[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                igx[j] = grp_table[(size_t)i * 8 + j];
                igy[j] = grp_table[(size_t)i * 8 + 4 + j];
            }

            ulong inv_i[4];
            mod_mul(inv_i, acc, pre[i - 1]);
            ulong den_i[4];
            mod_sub(den_i, igx, Cx);
            mod_mul(acc, acc, den_i);

            // C + i*G  -> covers index GROUP_H + i
            ulong px[4], py[4];
            affine_add(Cx, Cy, igx, igy, inv_i, px, py);
            if (point_matches(px, py, group_base + GROUP_H + i, total_keys,
                              start_lo, start_hi, target_h160, match_lo,
                              match_hi, match_found))
                return;

            // C - i*G  -> covers index GROUP_H - i (same denominator)
            ulong negy[4];
            mod_sub(negy, PM, igy);
            affine_add(Cx, Cy, igx, negy, inv_i, px, py);
            if (point_matches(px, py, group_base + GROUP_H - i, total_keys,
                              start_lo, start_hi, target_h160, match_lo,
                              match_hi, match_found))
                return;
        }

        // The centre itself.
        if (point_matches(Cx, Cy, group_base + GROUP_H, total_keys, start_lo,
                          start_hi, target_h160, match_lo, match_hi, match_found))
            return;

        // ---- advance the centre: C += GROUP_SIZE*G ----
        // acc now holds 1/den[0], exactly the slope denominator needed here.
        {
            ulong nx[4], ny[4];
            affine_add(Cx, Cy, step_x, step_y, acc, nx, ny);
            fcopy(Cx, nx);
            fcopy(Cy, ny);
        }
    }
}
