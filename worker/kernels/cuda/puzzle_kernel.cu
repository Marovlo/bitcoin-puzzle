/**
 * Bitcoin Puzzle CUDA Kernel - Optimized for NVIDIA GPUs
 *
 * Key advantages over Metal:
 *   - Native __umul64hi: 64x64->128 in ONE instruction (vs 4x mul32 on Metal)
 *   - Larger register file: mod_inv's 12 temp vars fit without spilling
 *   - Higher occupancy: more warps hide memory latency
 *
 * Per thread: scalar_mul_g (windowed) → mod_inv → compress → sha256 → ripemd160 → compare
 */

#include <cstdint>
#include <cstdio>

// ========== 256-bit field arithmetic using native 64-bit mul ==========

__device__ __forceinline__
uint64_t addc(uint64_t a, uint64_t b, uint64_t &carry) {
    uint64_t r;
    asm("add.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
    asm("addc.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(r), "l"(carry));
    asm("addc.u64 %0, 0, 0;" : "=l"(carry));
    return r;
}

__device__ __forceinline__
uint64_t subb(uint64_t a, uint64_t b, uint64_t &borrow) {
    uint64_t r;
    asm("sub.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
    asm("subc.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(r), "l"(borrow));
    asm("subc.u64 %0, 0, 0;" : "=l"(borrow));
    // borrow is 0xFFFFFFFFFFFFFFFF if underflow, but we want 0 or 1
    borrow = borrow ? 1 : 0;
    return r;
}

// Native 64x64 → 128 using PTX
__device__ __forceinline__
uint64_t mul128(uint64_t a, uint64_t b, uint64_t &hi) {
    uint64_t lo;
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
    return lo;
}

// secp256k1 prime constants
__device__ static const uint64_t P0 = 0xFFFFFFFEFFFFFC2Full;
__device__ static const uint64_t P1 = 0xFFFFFFFFFFFFFFFFull;
__device__ static const uint64_t P2 = 0xFFFFFFFFFFFFFFFFull;
__device__ static const uint64_t P3 = 0xFFFFFFFFFFFFFFFFull;
__device__ static const uint64_t K0 = 0x1000003D1ull;

// ========== mod_add / mod_sub ==========

__device__ void mod_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t c = 0;
    r[0] = addc(a[0], b[0], c); r[1] = addc(a[1], b[1], c);
    r[2] = addc(a[2], b[2], c); r[3] = addc(a[3], b[3], c);
    if (c) { uint64_t cc = 0; r[0] = addc(r[0], K0, cc); r[1] = addc(r[1], 0, cc);
             r[2] = addc(r[2], 0, cc); r[3] = addc(r[3], 0, cc); }
    bool ge = (r[3] > P3) || (r[3] == P3 && r[2] > P2) ||
              (r[3] == P3 && r[2] == P2 && r[1] > P1) ||
              (r[3] == P3 && r[2] == P2 && r[1] == P1 && r[0] >= P0);
    if (ge) { uint64_t bo = 0; r[0] = subb(r[0], P0, bo); r[1] = subb(r[1], P1, bo);
              r[2] = subb(r[2], P2, bo); r[3] = subb(r[3], P3, bo); }
}

__device__ void mod_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t bo = 0;
    r[0] = subb(a[0], b[0], bo); r[1] = subb(a[1], b[1], bo);
    r[2] = subb(a[2], b[2], bo); r[3] = subb(a[3], b[3], bo);
    if (bo) { uint64_t c = 0; r[0] = addc(r[0], P0, c); r[1] = addc(r[1], P1, c);
              r[2] = addc(r[2], P2, c); r[3] = addc(r[3], P3, c); }
}

// ========== mod_mul: full 256x256 → 512, reduce mod p ==========

__device__ void mod_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t w[8] = {0,0,0,0,0,0,0,0};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo = mul128(a[i], b[j], hi);
            uint64_t c1 = 0, c2 = 0;
            uint64_t s0 = addc(w[i+j], lo, c1);
            uint64_t s1 = addc(s0, carry, c2);
            w[i+j] = s1;
            carry = hi + c1 + c2;
        }
        w[i+4] = carry;
    }

    // Reduce: hi * K0 + lo
    uint64_t h[5] = {0,0,0,0,0};
    {   uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t hh, ll = mul128(w[4+i], K0, hh);
            uint64_t c1 = 0, c2 = 0;
            uint64_t s0 = addc(h[i], ll, c1);
            uint64_t s1 = addc(s0, carry, c2);
            h[i] = s1;
            carry = hh + c1 + c2;
        }
        h[4] = carry;
    }

    uint64_t c = 0;
    uint64_t t0 = addc(w[0], h[0], c); uint64_t t1 = addc(w[1], h[1], c);
    uint64_t t2 = addc(w[2], h[2], c); uint64_t t3 = addc(w[3], h[3], c);
    uint64_t overflow = c + h[4];

    if (overflow) {
        uint64_t oh, ol = mul128(overflow, K0, oh);
        uint64_t c2 = 0;
        t0 = addc(t0, ol, c2); t1 = addc(t1, oh, c2);
        t2 = addc(t2, 0, c2); t3 = addc(t3, 0, c2);
        if (c2) { uint64_t c3 = 0; t0 = addc(t0, K0, c3); t1 = addc(t1, 0, c3);
                  t2 = addc(t2, 0, c3); t3 = addc(t3, 0, c3); }
    }

    bool ge = (t3 > P3) || (t3 == P3 && t2 > P2) ||
              (t3 == P3 && t2 == P2 && t1 > P1) ||
              (t3 == P3 && t2 == P2 && t1 == P1 && t0 >= P0);
    if (ge) { uint64_t bo = 0; t0 = subb(t0, P0, bo); t1 = subb(t1, P1, bo);
              t2 = subb(t2, P2, bo); t3 = subb(t3, P3, bo); }
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

__device__ __forceinline__ void mod_sqr(uint64_t r[4], const uint64_t a[4]) {
    mod_mul(r, a, a);
}

// ========== mod_inv: a^(p-2) mod p (addition chain) ==========

__device__ void mod_inv(uint64_t r[4], const uint64_t a[4]) {
    uint64_t x2[4], x3[4], x6[4], x9[4], x11[4], x22[4], x44[4];
    uint64_t x88[4], x176[4], x220[4], x223[4], t[4];

    mod_sqr(x2, a); mod_mul(x2, x2, a);
    mod_sqr(x3, x2); mod_mul(x3, x3, a);
    mod_sqr(x6, x3); mod_sqr(x6, x6); mod_sqr(x6, x6); mod_mul(x6, x6, x3);
    mod_sqr(x9, x6); mod_sqr(x9, x9); mod_sqr(x9, x9); mod_mul(x9, x9, x3);
    mod_sqr(x11, x9); mod_sqr(x11, x11); mod_mul(x11, x11, x2);

    #pragma unroll
    for (int i = 0; i < 4; i++) x22[i] = x11[i];
    for (int i = 0; i < 11; i++) mod_sqr(x22, x22);
    mod_mul(x22, x22, x11);

    #pragma unroll
    for (int i = 0; i < 4; i++) x44[i] = x22[i];
    for (int i = 0; i < 22; i++) mod_sqr(x44, x44);
    mod_mul(x44, x44, x22);

    #pragma unroll
    for (int i = 0; i < 4; i++) x88[i] = x44[i];
    for (int i = 0; i < 44; i++) mod_sqr(x88, x88);
    mod_mul(x88, x88, x44);

    #pragma unroll
    for (int i = 0; i < 4; i++) x176[i] = x88[i];
    for (int i = 0; i < 88; i++) mod_sqr(x176, x176);
    mod_mul(x176, x176, x88);

    #pragma unroll
    for (int i = 0; i < 4; i++) x220[i] = x176[i];
    for (int i = 0; i < 44; i++) mod_sqr(x220, x220);
    mod_mul(x220, x220, x44);

    mod_sqr(x223, x220); mod_sqr(x223, x223); mod_sqr(x223, x223);
    mod_mul(x223, x223, x3);

    #pragma unroll
    for (int i = 0; i < 4; i++) t[i] = x223[i];
    for (int i = 0; i < 23; i++) mod_sqr(t, t);
    mod_mul(t, t, x22);
    for (int i = 0; i < 5; i++) mod_sqr(t, t);
    mod_mul(t, t, a);
    for (int i = 0; i < 3; i++) mod_sqr(t, t);
    mod_mul(t, t, x2);
    mod_sqr(t, t); mod_sqr(t, t);
    mod_mul(t, t, a);

    r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
}

// ========== EC point operations ==========

__device__ __forceinline__ bool limbs_zero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

__device__ void jac_double(uint64_t rx[4], uint64_t ry[4], uint64_t rz[4],
                           const uint64_t px[4], const uint64_t py[4], const uint64_t pz[4]) {
    if (limbs_zero(pz) || limbs_zero(py)) {
        memset(rx, 0, 32); memset(ry, 0, 32); memset(rz, 0, 32); return;
    }
    uint64_t y2[4], s[4], m[4], t[4], y4[4];
    mod_sqr(y2, py); mod_mul(s, px, y2); mod_add(s, s, s); mod_add(s, s, s);
    mod_sqr(m, px); mod_add(t, m, m); mod_add(m, t, m);
    mod_sqr(rx, m); mod_sub(rx, rx, s); mod_sub(rx, rx, s);
    mod_sub(t, s, rx); mod_mul(t, m, t);
    mod_sqr(y4, y2); mod_add(y4, y4, y4); mod_add(y4, y4, y4); mod_add(y4, y4, y4);
    mod_sub(ry, t, y4);
    mod_mul(rz, py, pz); mod_add(rz, rz, rz);
}

__device__ void jac_add_mixed(uint64_t rx[4], uint64_t ry[4], uint64_t rz[4],
                              const uint64_t px[4], const uint64_t py[4], const uint64_t pz[4],
                              const uint64_t qx[4], const uint64_t qy[4]) {
    if (limbs_zero(qx) && limbs_zero(qy)) {
        memcpy(rx, px, 32); memcpy(ry, py, 32); memcpy(rz, pz, 32); return;
    }
    if (limbs_zero(pz)) {
        memcpy(rx, qx, 32); memcpy(ry, qy, 32);
        rz[0] = 1; rz[1] = 0; rz[2] = 0; rz[3] = 0; return;
    }
    uint64_t z2[4], z3[4], u2[4], s2[4], h[4], rr[4], h2[4], h3[4], v[4], tmp[4];
    mod_sqr(z2, pz); mod_mul(z3, z2, pz); mod_mul(u2, qx, z2); mod_mul(s2, qy, z3);
    mod_sub(h, u2, px); mod_sub(rr, s2, py);
    if (limbs_zero(h)) {
        if (limbs_zero(rr)) { jac_double(rx, ry, rz, px, py, pz); }
        else { memset(rx, 0, 32); memset(ry, 0, 32); memset(rz, 0, 32); }
        return;
    }
    mod_sqr(h2, h); mod_mul(h3, h2, h); mod_mul(v, px, h2);
    mod_sqr(tmp, rr); mod_sub(tmp, tmp, h3); mod_sub(tmp, tmp, v); mod_sub(rx, tmp, v);
    mod_sub(tmp, v, rx); mod_mul(tmp, rr, tmp);
    uint64_t yh3[4]; mod_mul(yh3, py, h3); mod_sub(ry, tmp, yh3);
    mod_mul(rz, pz, h);
}

// Windowed scalar*G multiplication (8-bit windows, 32 windows)
__device__ void scalar_mul_g(uint64_t rx[4], uint64_t ry[4], uint64_t rz[4],
                             const uint64_t scalar[4], const uint64_t* __restrict__ g_table) {
    memset(rx, 0, 32); memset(ry, 0, 32); memset(rz, 0, 32);

    #pragma unroll
    for (uint32_t w = 0; w < 32; w++) {
        uint32_t limb_idx = w >> 3;
        uint32_t byte_in_limb = w & 7;
        uint32_t byte_val = (uint32_t)((scalar[limb_idx] >> (8 * byte_in_limb)) & 0xFF);
        if (byte_val == 0) continue;

        uint32_t base = (w * 256 + byte_val) * 8;
        const uint64_t* qx = &g_table[base];
        const uint64_t* qy = &g_table[base + 4];

        uint64_t nx[4], ny[4], nz[4];
        jac_add_mixed(nx, ny, nz, rx, ry, rz, qx, qy);
        memcpy(rx, nx, 32); memcpy(ry, ny, 32); memcpy(rz, nz, 32);
    }
}

// ========== SHA-256 (33 bytes) ==========

__device__ static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ void sha256_33(const uint8_t pk[33], uint8_t hash[32]) {
    uint32_t W[64];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        W[i] = ((uint32_t)pk[i*4]<<24)|((uint32_t)pk[i*4+1]<<16)|((uint32_t)pk[i*4+2]<<8)|(uint32_t)pk[i*4+3];
    W[8] = ((uint32_t)pk[32]<<24)|(0x80u<<16);
    W[9]=0; W[10]=0; W[11]=0; W[12]=0; W[13]=0; W[14]=0; W[15]=264;

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(W[i-15],7)^rotr32(W[i-15],18)^(W[i-15]>>3);
        uint32_t s1 = rotr32(W[i-2],17)^rotr32(W[i-2],19)^(W[i-2]>>10);
        W[i] = W[i-16]+s0+W[i-7]+s1;
    }

    uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a;
    uint32_t e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        uint32_t ch=(e&f)^(~e&g);
        uint32_t t1=h+S1+ch+SHA256_K[i]+W[i];
        uint32_t S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    a+=0x6a09e667;b+=0xbb67ae85;c+=0x3c6ef372;d+=0xa54ff53a;
    e+=0x510e527f;f+=0x9b05688c;g+=0x1f83d9ab;h+=0x5be0cd19;

    uint32_t H[8]={a,b,c,d,e,f,g,h};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i*4]=(uint8_t)(H[i]>>24); hash[i*4+1]=(uint8_t)(H[i]>>16);
        hash[i*4+2]=(uint8_t)(H[i]>>8); hash[i*4+3]=(uint8_t)H[i];
    }
}

// ========== RIPEMD-160 (32 bytes) ==========

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

__device__ static const uint32_t RKL[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
__device__ static const uint32_t RKR[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
__device__ static const int RRL[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
__device__ static const int RRR[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
__device__ static const int RSL[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
__device__ static const int RSR[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};

__device__ void ripemd160_32(const uint8_t sha[32], uint8_t h160[20]) {
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        X[i]=(uint32_t)sha[i*4]|((uint32_t)sha[i*4+1]<<8)|((uint32_t)sha[i*4+2]<<16)|((uint32_t)sha[i*4+3]<<24);
    X[8]=0x80u; for (int i=9;i<14;i++) X[i]=0; X[14]=256u; X[15]=0;

    uint32_t AL=0x67452301,BL=0xEFCDAB89,CL=0x98BADCFE,DL=0x10325476,EL=0xC3D2E1F0;
    uint32_t AR=AL,BR=BL,CR=CL,DR=DL,ER=EL;

    for (int j=0;j<16;j++){uint32_t tL=rotl32(AL+(BL^CL^DL)+X[RRL[j]]+RKL[0],RSL[j])+EL;AL=EL;EL=DL;DL=rotl32(CL,10);CL=BL;BL=tL;uint32_t tR=rotl32(AR+(BR^(CR|~DR))+X[RRR[j]]+RKR[0],RSR[j])+ER;AR=ER;ER=DR;DR=rotl32(CR,10);CR=BR;BR=tR;}
    for (int j=16;j<32;j++){uint32_t tL=rotl32(AL+((BL&CL)|(~BL&DL))+X[RRL[j]]+RKL[1],RSL[j])+EL;AL=EL;EL=DL;DL=rotl32(CL,10);CL=BL;BL=tL;uint32_t tR=rotl32(AR+((BR&DR)|(CR&~DR))+X[RRR[j]]+RKR[1],RSR[j])+ER;AR=ER;ER=DR;DR=rotl32(CR,10);CR=BR;BR=tR;}
    for (int j=32;j<48;j++){uint32_t tL=rotl32(AL+((BL|~CL)^DL)+X[RRL[j]]+RKL[2],RSL[j])+EL;AL=EL;EL=DL;DL=rotl32(CL,10);CL=BL;BL=tL;uint32_t tR=rotl32(AR+((BR|~CR)^DR)+X[RRR[j]]+RKR[2],RSR[j])+ER;AR=ER;ER=DR;DR=rotl32(CR,10);CR=BR;BR=tR;}
    for (int j=48;j<64;j++){uint32_t tL=rotl32(AL+((BL&DL)|(CL&~DL))+X[RRL[j]]+RKL[3],RSL[j])+EL;AL=EL;EL=DL;DL=rotl32(CL,10);CL=BL;BL=tL;uint32_t tR=rotl32(AR+((BR&CR)|(~BR&DR))+X[RRR[j]]+RKR[3],RSR[j])+ER;AR=ER;ER=DR;DR=rotl32(CR,10);CR=BR;BR=tR;}
    for (int j=64;j<80;j++){uint32_t tL=rotl32(AL+(BL^(CL|~DL))+X[RRL[j]]+RKL[4],RSL[j])+EL;AL=EL;EL=DL;DL=rotl32(CL,10);CL=BL;BL=tL;uint32_t tR=rotl32(AR+(BR^CR^DR)+X[RRR[j]]+RKR[4],RSR[j])+ER;AR=ER;ER=DR;DR=rotl32(CR,10);CR=BR;BR=tR;}

    uint32_t H0=0x67452301,H1=0xEFCDAB89,H2=0x98BADCFE,H3=0x10325476,H4=0xC3D2E1F0;
    uint32_t t=H1+CL+DR; H1=H2+DL+ER; H2=H3+EL+AR; H3=H4+AL+BR; H4=H0+BL+CR; H0=t;
    uint32_t Hf[5]={H0,H1,H2,H3,H4};
    #pragma unroll
    for (int i=0;i<5;i++){h160[i*4]=(uint8_t)Hf[i];h160[i*4+1]=(uint8_t)(Hf[i]>>8);h160[i*4+2]=(uint8_t)(Hf[i]>>16);h160[i*4+3]=(uint8_t)(Hf[i]>>24);}
}

// ========== Main Kernel ==========

extern "C" __global__ void puzzle_search(
    const uint64_t* __restrict__ g_table,    // precomputed G table
    const uint8_t*  __restrict__ target_h160, // 20 bytes
    uint64_t start_lo,
    uint64_t start_hi,
    uint64_t total_keys,
    uint64_t* match_lo,
    uint64_t* match_hi,
    uint32_t* match_found)
{
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_keys) return;
    if (atomicAdd(match_found, 0) != 0) return; // early exit if found

    // k = start + gid
    uint64_t k[4];
    uint64_t c = 0;
    k[0] = addc(start_lo, gid, c);
    k[1] = addc(start_hi, 0, c);
    k[2] = c;
    k[3] = 0;

    // P = k * G
    uint64_t px[4], py[4], pz[4];
    scalar_mul_g(px, py, pz, k, g_table);
    if (limbs_zero(pz)) return;

    // Affinize
    uint64_t zi[4], zi2[4], zi3[4], ax[4], ay[4];
    mod_inv(zi, pz);
    mod_sqr(zi2, zi);
    mod_mul(zi3, zi2, zi);
    mod_mul(ax, px, zi2);
    mod_mul(ay, py, zi3);

    // Compress pubkey
    uint8_t pubkey[33];
    pubkey[0] = 0x02 | (uint8_t)(ay[0] & 1);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t l = ax[3-i];
        #pragma unroll
        for (int j = 0; j < 8; j++)
            pubkey[1+i*8+j] = (uint8_t)(l >> (56-8*j));
    }

    // Hash
    uint8_t sha[32], h160[20];
    sha256_33(pubkey, sha);
    ripemd160_32(sha, h160);

    // Compare
    bool match = true;
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        if (h160[i] != target_h160[i]) { match = false; break; }
    }

    if (match) {
        if (atomicCAS(match_found, 0u, 1u) == 0u) {
            match_lo[0] = k[0];
            match_hi[0] = k[1];
        }
    }
}
