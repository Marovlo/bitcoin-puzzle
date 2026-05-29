#pragma once
#include "types.h"
#include <cstring>

namespace secp256k1 {

// secp256k1 prime: p = 2^256 - 2^32 - 977
static constexpr uint64_t P[4] = {
    0xFFFFFFFEFFFFFC2Full, 0xFFFFFFFFFFFFFFFFull,
    0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull
};
// Reduction constant: 2^256 mod p = 2^32 + 977
static constexpr uint64_t K0 = 0x1000003D1ull;

// Generator point G (affine)
static constexpr uint64_t GX[4] = {
    0x59F2815B16F81798ull, 0x029BFCDB2DCE28D9ull,
    0x55A06295CE870B07ull, 0x79BE667EF9DCBBACull
};
static constexpr uint64_t GY[4] = {
    0x9C47D08FFB10D4B8ull, 0xFD17B448A6855419ull,
    0x5DA4FBFC0E1108A8ull, 0x483ADA7726A3C465ull
};

// --- 256-bit arithmetic with carry/borrow ---

inline uint64_t addc(uint64_t a, uint64_t b, uint64_t& carry) {
    unsigned __int128 s = (unsigned __int128)a + b + carry;
    carry = (uint64_t)(s >> 64);
    return (uint64_t)s;
}

inline uint64_t subb(uint64_t a, uint64_t b, uint64_t& borrow) {
    unsigned __int128 s = (unsigned __int128)a - b - borrow;
    borrow = (s >> 127) ? 1 : 0;
    return (uint64_t)s;
}

inline void mod_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t c = 0;
    r[0] = addc(a[0], b[0], c);
    r[1] = addc(a[1], b[1], c);
    r[2] = addc(a[2], b[2], c);
    r[3] = addc(a[3], b[3], c);
    if (c) {
        uint64_t cc = 0;
        r[0] = addc(r[0], K0, cc);
        r[1] = addc(r[1], 0, cc);
        r[2] = addc(r[2], 0, cc);
        r[3] = addc(r[3], 0, cc);
    }
    // Reduce if >= p
    bool ge = (r[3] > P[3]) ||
              (r[3] == P[3] && r[2] > P[2]) ||
              (r[3] == P[3] && r[2] == P[2] && r[1] > P[1]) ||
              (r[3] == P[3] && r[2] == P[2] && r[1] == P[1] && r[0] >= P[0]);
    if (ge) {
        uint64_t bo = 0;
        r[0] = subb(r[0], P[0], bo);
        r[1] = subb(r[1], P[1], bo);
        r[2] = subb(r[2], P[2], bo);
        r[3] = subb(r[3], P[3], bo);
    }
}

inline void mod_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t bo = 0;
    r[0] = subb(a[0], b[0], bo);
    r[1] = subb(a[1], b[1], bo);
    r[2] = subb(a[2], b[2], bo);
    r[3] = subb(a[3], b[3], bo);
    if (bo) {
        uint64_t c = 0;
        r[0] = addc(r[0], P[0], c);
        r[1] = addc(r[1], P[1], c);
        r[2] = addc(r[2], P[2], c);
        r[3] = addc(r[3], P[3], c);
    }
}

// 256x256 -> 512, then reduce mod p using 2^256 = K0 trick
inline void mod_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    // Full 512-bit product via schoolbook with 128-bit intermediates
    unsigned __int128 w[8] = {};
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; ++j) {
            unsigned __int128 prod = (unsigned __int128)a[i] * b[j] + w[i+j] + carry;
            w[i+j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        w[i+4] = carry;
    }

    uint64_t lo[4] = {(uint64_t)w[0], (uint64_t)w[1], (uint64_t)w[2], (uint64_t)w[3]};
    uint64_t hi[4] = {(uint64_t)w[4], (uint64_t)w[5], (uint64_t)w[6], (uint64_t)w[7]};

    // Reduce: result = lo + hi * K0
    unsigned __int128 carry = 0;
    uint64_t h[5];
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 prod = (unsigned __int128)hi[i] * K0 + carry;
        h[i] = (uint64_t)prod;
        carry = prod >> 64;
    }
    h[4] = (uint64_t)carry;

    uint64_t c = 0;
    r[0] = addc(lo[0], h[0], c);
    r[1] = addc(lo[1], h[1], c);
    r[2] = addc(lo[2], h[2], c);
    r[3] = addc(lo[3], h[3], c);
    uint64_t overflow = c + h[4];

    if (overflow) {
        unsigned __int128 ov = (unsigned __int128)overflow * K0;
        uint64_t c2 = 0;
        r[0] = addc(r[0], (uint64_t)ov, c2);
        r[1] = addc(r[1], (uint64_t)(ov >> 64), c2);
        r[2] = addc(r[2], 0, c2);
        r[3] = addc(r[3], 0, c2);
        if (c2) {
            uint64_t c3 = 0;
            r[0] = addc(r[0], K0, c3);
            r[1] = addc(r[1], 0, c3);
            r[2] = addc(r[2], 0, c3);
            r[3] = addc(r[3], 0, c3);
        }
    }

    bool ge = (r[3] > P[3]) ||
              (r[3] == P[3] && r[2] > P[2]) ||
              (r[3] == P[3] && r[2] == P[2] && r[1] > P[1]) ||
              (r[3] == P[3] && r[2] == P[2] && r[1] == P[1] && r[0] >= P[0]);
    if (ge) {
        uint64_t bo = 0;
        r[0] = subb(r[0], P[0], bo);
        r[1] = subb(r[1], P[1], bo);
        r[2] = subb(r[2], P[2], bo);
        r[3] = subb(r[3], P[3], bo);
    }
}

inline void mod_sqr(uint64_t r[4], const uint64_t a[4]) {
    mod_mul(r, a, a);
}

// Modular inverse: a^(p-2) mod p using libsecp256k1 addition chain
inline void mod_inv(uint64_t r[4], const uint64_t a[4]) {
    uint64_t x2[4], x3[4], x6[4], x9[4], x11[4], x22[4], x44[4];
    uint64_t x88[4], x176[4], x220[4], x223[4], t[4];

    mod_sqr(x2, a); mod_mul(x2, x2, a);
    mod_sqr(x3, x2); mod_mul(x3, x3, a);
    mod_sqr(x6, x3); mod_sqr(x6, x6); mod_sqr(x6, x6); mod_mul(x6, x6, x3);
    mod_sqr(x9, x6); mod_sqr(x9, x9); mod_sqr(x9, x9); mod_mul(x9, x9, x3);
    mod_sqr(x11, x9); mod_sqr(x11, x11); mod_mul(x11, x11, x2);

    memcpy(x22, x11, 32);
    for (int i = 0; i < 11; ++i) mod_sqr(x22, x22);
    mod_mul(x22, x22, x11);

    memcpy(x44, x22, 32);
    for (int i = 0; i < 22; ++i) mod_sqr(x44, x44);
    mod_mul(x44, x44, x22);

    memcpy(x88, x44, 32);
    for (int i = 0; i < 44; ++i) mod_sqr(x88, x88);
    mod_mul(x88, x88, x44);

    memcpy(x176, x88, 32);
    for (int i = 0; i < 88; ++i) mod_sqr(x176, x176);
    mod_mul(x176, x176, x88);

    memcpy(x220, x176, 32);
    for (int i = 0; i < 44; ++i) mod_sqr(x220, x220);
    mod_mul(x220, x220, x44);

    mod_sqr(x223, x220); mod_sqr(x223, x223); mod_sqr(x223, x223);
    mod_mul(x223, x223, x3);

    memcpy(t, x223, 32);
    for (int i = 0; i < 23; ++i) mod_sqr(t, t);
    mod_mul(t, t, x22);
    for (int i = 0; i < 5; ++i) mod_sqr(t, t);
    mod_mul(t, t, a);
    for (int i = 0; i < 3; ++i) mod_sqr(t, t);
    mod_mul(t, t, x2);
    mod_sqr(t, t); mod_sqr(t, t);
    mod_mul(t, t, a);

    memcpy(r, t, 32);
}

// --- Jacobian point operations ---
struct JacobianPoint {
    uint64_t X[4], Y[4], Z[4];
};

inline bool is_infinity(const JacobianPoint& p) {
    return p.Z[0] == 0 && p.Z[1] == 0 && p.Z[2] == 0 && p.Z[3] == 0;
}

inline void point_double(JacobianPoint& r, const JacobianPoint& p) {
    if (is_infinity(p) || (p.Y[0] == 0 && p.Y[1] == 0 && p.Y[2] == 0 && p.Y[3] == 0)) {
        memset(&r, 0, sizeof(r));
        return;
    }
    uint64_t y2[4], s[4], m[4], t[4], y4[4];
    mod_sqr(y2, p.Y);
    mod_mul(s, p.X, y2);
    mod_add(s, s, s); mod_add(s, s, s);  // S = 4*X*Y^2
    mod_sqr(m, p.X);
    mod_add(t, m, m); mod_add(m, t, m);  // M = 3*X^2 (a=0 for secp256k1)

    mod_sqr(r.X, m);
    mod_sub(r.X, r.X, s); mod_sub(r.X, r.X, s);  // X' = M^2 - 2S

    mod_sub(t, s, r.X);
    mod_mul(t, m, t);
    mod_sqr(y4, y2);
    mod_add(y4, y4, y4); mod_add(y4, y4, y4); mod_add(y4, y4, y4);  // 8*Y^4
    mod_sub(r.Y, t, y4);

    mod_mul(r.Z, p.Y, p.Z);
    mod_add(r.Z, r.Z, r.Z);
}

// Mixed addition: P (Jacobian) + Q (affine)
inline void point_add_mixed(JacobianPoint& r, const JacobianPoint& p,
                            const uint64_t qx[4], const uint64_t qy[4]) {
    if (qx[0] == 0 && qx[1] == 0 && qx[2] == 0 && qx[3] == 0 &&
        qy[0] == 0 && qy[1] == 0 && qy[2] == 0 && qy[3] == 0) {
        r = p; return;
    }
    if (is_infinity(p)) {
        memcpy(r.X, qx, 32); memcpy(r.Y, qy, 32);
        r.Z[0] = 1; r.Z[1] = 0; r.Z[2] = 0; r.Z[3] = 0;
        return;
    }
    uint64_t z2[4], z3[4], u2[4], s2[4], h[4], rr[4], h2[4], h3[4], v[4], tmp[4];
    mod_sqr(z2, p.Z);
    mod_mul(z3, z2, p.Z);
    mod_mul(u2, qx, z2);
    mod_mul(s2, qy, z3);
    mod_sub(h, u2, p.X);
    mod_sub(rr, s2, p.Y);

    if (h[0] == 0 && h[1] == 0 && h[2] == 0 && h[3] == 0) {
        if (rr[0] == 0 && rr[1] == 0 && rr[2] == 0 && rr[3] == 0) {
            point_double(r, p);
        } else {
            memset(&r, 0, sizeof(r));
        }
        return;
    }

    mod_sqr(h2, h);
    mod_mul(h3, h2, h);
    mod_mul(v, p.X, h2);
    mod_sqr(tmp, rr);
    mod_sub(tmp, tmp, h3);
    mod_sub(tmp, tmp, v);
    mod_sub(r.X, tmp, v);

    mod_sub(tmp, v, r.X);
    mod_mul(tmp, rr, tmp);
    uint64_t yh3[4];
    mod_mul(yh3, p.Y, h3);
    mod_sub(r.Y, tmp, yh3);
    mod_mul(r.Z, p.Z, h);
}

// Precomputed G table: 32 windows x 256 entries, each entry = affine (X,Y)
static constexpr size_t G_TABLE_WINDOWS = 32;
static constexpr size_t G_TABLE_ENTRIES = 256;
static constexpr size_t G_TABLE_ULONGS = G_TABLE_WINDOWS * G_TABLE_ENTRIES * 8;

inline void build_g_table(uint64_t* table) {
    memset(table, 0, G_TABLE_ULONGS * sizeof(uint64_t));

    for (uint32_t w = 0; w < G_TABLE_WINDOWS; ++w) {
        // base_scalar = 2^(8*w)
        uint256_t base_scalar{};
        uint32_t bit = 8u * w;
        base_scalar.d[bit / 64] = (uint64_t)1 << (bit % 64);

        // Compute base_point = base_scalar * G using double-and-add
        JacobianPoint base_jac{};
        memcpy(base_jac.X, GX, 32);
        memcpy(base_jac.Y, GY, 32);
        base_jac.Z[0] = 1;

        // scalar mul: base_scalar * G
        JacobianPoint result{};
        for (int b = 255; b >= 0; --b) {
            JacobianPoint dbl;
            point_double(dbl, result);
            result = dbl;
            uint64_t limb = base_scalar.d[b / 64];
            if ((limb >> (b % 64)) & 1) {
                JacobianPoint add;
                point_add_mixed(add, result, GX, GY);
                result = add;
            }
        }

        // Convert to affine for the first entry
        uint64_t inv_z[4], inv_z2[4], inv_z3[4], ax[4], ay[4];
        if (!is_infinity(result)) {
            mod_inv(inv_z, result.Z);
            mod_sqr(inv_z2, inv_z);
            mod_mul(inv_z3, inv_z2, inv_z);
            mod_mul(ax, result.X, inv_z2);
            mod_mul(ay, result.Y, inv_z3);
        } else {
            memset(ax, 0, 32); memset(ay, 0, 32);
        }

        // Now build entries incrementally: entry[d] = d * base_point
        // entry[1] = base_point (already computed as ax, ay)
        // entry[d] = entry[d-1] + base_point
        JacobianPoint acc{};
        memcpy(acc.X, ax, 32);
        memcpy(acc.Y, ay, 32);
        acc.Z[0] = 1; acc.Z[1] = 0; acc.Z[2] = 0; acc.Z[3] = 0;

        for (uint32_t d = 1; d < G_TABLE_ENTRIES; ++d) {
            // Store current acc in affine
            uint64_t entry_x[4], entry_y[4];
            if (!is_infinity(acc)) {
                mod_inv(inv_z, acc.Z);
                mod_sqr(inv_z2, inv_z);
                mod_mul(inv_z3, inv_z2, inv_z);
                mod_mul(entry_x, acc.X, inv_z2);
                mod_mul(entry_y, acc.Y, inv_z3);
            } else {
                memset(entry_x, 0, 32);
                memset(entry_y, 0, 32);
            }

            size_t base_idx = (size_t)(w * G_TABLE_ENTRIES + d) * 8;
            memcpy(&table[base_idx], entry_x, 32);
            memcpy(&table[base_idx + 4], entry_y, 32);

            if (d + 1 < G_TABLE_ENTRIES) {
                JacobianPoint next;
                point_add_mixed(next, acc, ax, ay);
                acc = next;
            }
        }
    }
}

// Windowed scalar multiplication using precomputed G table
inline void scalar_mul_g_windowed(JacobianPoint& r, const uint64_t scalar[4],
                                  const uint64_t* g_table) {
    memset(&r, 0, sizeof(r));  // Start at infinity

    for (uint32_t w = 0; w < 32; ++w) {
        uint32_t limb_idx = w >> 3;
        uint32_t byte_in_limb = w & 7;
        uint32_t byte_val = (uint32_t)((scalar[limb_idx] >> (8 * byte_in_limb)) & 0xFF);

        if (byte_val == 0) continue;

        size_t base_idx = (size_t)(w * 256 + byte_val) * 8;
        const uint64_t* qx = &g_table[base_idx];
        const uint64_t* qy = &g_table[base_idx + 4];

        JacobianPoint next;
        point_add_mixed(next, r, qx, qy);
        r = next;
    }
}

}  // namespace secp256k1
