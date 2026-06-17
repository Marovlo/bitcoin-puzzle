/*
 * Bitcoin Puzzle Brute-Force Metal Kernel
 *
 * Per thread pipeline:
 *   1. k = base + gid
 *   2. P = k * G (8-bit windowed precomputed table)
 *   3. Affinize P (one mod_inv per thread)
 *   4. Compress pubkey -> 33 bytes
 *   5. SHA-256 (single 64-byte block)
 *   6. RIPEMD-160 (single 64-byte block)
 *   7. Compare to target hash160
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// === 128-bit arithmetic helpers ===

inline ulong addc(ulong a, ulong b, thread ulong &carry) {
    ulong sum  = a + b;
    ulong c1   = (sum < a) ? 1ul : 0ul;
    ulong sum2 = sum + carry;
    ulong c2   = (sum2 < sum) ? 1ul : 0ul;
    carry = c1 + c2;
    return sum2;
}

inline ulong subb(ulong a, ulong b, thread ulong &borrow) {
    ulong diff  = a - b;
    ulong b1    = (a < b) ? 1ul : 0ul;
    ulong diff2 = diff - borrow;
    ulong b2    = (diff < borrow) ? 1ul : 0ul;
    borrow = b1 + b2;
    return diff2;
}

inline ulong mul128(ulong a, ulong b, thread ulong &hi) {
    ulong al = a & 0xFFFFFFFFul, ah = a >> 32;
    ulong bl = b & 0xFFFFFFFFul, bh = b >> 32;
    ulong p0 = al * bl;
    ulong p1 = al * bh;
    ulong p2 = ah * bl;
    ulong p3 = ah * bh;
    ulong mid = (p0 >> 32) + (p1 & 0xFFFFFFFFul) + (p2 & 0xFFFFFFFFul);
    ulong lo  = (p0 & 0xFFFFFFFFul) | (mid << 32);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    return lo;
}

// === secp256k1 constants ===
constant ulong P0 = 0xFFFFFFFEFFFFFC2Ful;
constant ulong P1 = 0xFFFFFFFFFFFFFFFFul;
constant ulong P2 = 0xFFFFFFFFFFFFFFFFul;
constant ulong P3 = 0xFFFFFFFFFFFFFFFFul;
constant ulong K0 = 0x00000001000003D1ul;

// Generator point G (affine), little-endian by limb. Used for the incremental
// step P_{i+1} = P_i + G when a thread walks consecutive keys.
constant ulong GX_[4] = {0x59F2815B16F81798ul, 0x029BFCDB2DCE28D9ul,
                         0x55A06295CE870B07ul, 0x79BE667EF9DCBBACul};
constant ulong GY_[4] = {0x9C47D08FFB10D4B8ul, 0xFD17B448A6855419ul,
                         0x5DA4FBFC0E1108A8ul, 0x483ADA7726A3C465ul};

// Keys processed per GPU thread. One full scalar_mul_g and one batched mod_inv
// are amortized across this many consecutive keys.
#define KEYS_PER_THREAD 128

// === Field arithmetic ===

inline void mod_add(thread ulong r[4], thread const ulong a[4], thread const ulong b[4]) {
    ulong c = 0;
    ulong t0 = addc(a[0], b[0], c);
    ulong t1 = addc(a[1], b[1], c);
    ulong t2 = addc(a[2], b[2], c);
    ulong t3 = addc(a[3], b[3], c);
    if (c) { ulong cc = 0; t0 = addc(t0, K0, cc); t1 = addc(t1, 0, cc); t2 = addc(t2, 0, cc); t3 = addc(t3, 0, cc); }
    bool ge = (t3 > P3) || (t3 == P3 && t2 > P2) || (t3 == P3 && t2 == P2 && t1 > P1) ||
              (t3 == P3 && t2 == P2 && t1 == P1 && t0 >= P0);
    if (ge) { ulong b2 = 0; t0 = subb(t0, P0, b2); t1 = subb(t1, P1, b2); t2 = subb(t2, P2, b2); t3 = subb(t3, P3, b2); }
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

inline void mod_sub(thread ulong r[4], thread const ulong a[4], thread const ulong b[4]) {
    ulong bo = 0;
    ulong t0 = subb(a[0], b[0], bo); ulong t1 = subb(a[1], b[1], bo);
    ulong t2 = subb(a[2], b[2], bo); ulong t3 = subb(a[3], b[3], bo);
    if (bo) { ulong c = 0; t0 = addc(t0, P0, c); t1 = addc(t1, P1, c); t2 = addc(t2, P2, c); t3 = addc(t3, P3, c); }
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

inline void mod_mul(thread ulong r[4], thread const ulong a[4], thread const ulong b[4]) {
    ulong w[8];
    for (int i = 0; i < 8; ++i) w[i] = 0;
    for (int i = 0; i < 4; ++i) {
        ulong carry = 0;
        for (int j = 0; j < 4; ++j) {
            ulong hi; ulong lo = mul128(a[i], b[j], hi);
            ulong c1 = 0, c2 = 0;
            ulong s0 = addc(w[i+j], lo, c1);
            ulong s1 = addc(s0, carry, c2);
            w[i+j] = s1; carry = hi + c1 + c2;
        }
        w[i+4] = carry;
    }
    ulong lo[4] = {w[0], w[1], w[2], w[3]};
    ulong hi[4] = {w[4], w[5], w[6], w[7]};
    ulong h[5] = {0,0,0,0,0};
    { ulong carry = 0;
      for (int i = 0; i < 4; ++i) { ulong hh; ulong ll = mul128(hi[i], K0, hh);
        ulong c1=0,c2=0; ulong s0=addc(h[i],ll,c1); ulong s1=addc(s0,carry,c2); h[i]=s1; carry=hh+c1+c2; }
      h[4]=carry; }
    ulong c = 0;
    ulong t0 = addc(lo[0], h[0], c); ulong t1 = addc(lo[1], h[1], c);
    ulong t2 = addc(lo[2], h[2], c); ulong t3 = addc(lo[3], h[3], c);
    ulong overflow = c + h[4];
    if (overflow) { ulong oh, ol; ol = mul128(overflow, K0, oh);
        ulong c2=0; t0=addc(t0,ol,c2); t1=addc(t1,oh,c2); t2=addc(t2,0,c2); t3=addc(t3,0,c2);
        if(c2){ulong c3=0;t0=addc(t0,K0,c3);t1=addc(t1,0,c3);t2=addc(t2,0,c3);t3=addc(t3,0,c3);} }
    bool ge = (t3>P3)||(t3==P3&&t2>P2)||(t3==P3&&t2==P2&&t1>P1)||(t3==P3&&t2==P2&&t1==P1&&t0>=P0);
    if (ge) { ulong bo=0; t0=subb(t0,P0,bo); t1=subb(t1,P1,bo); t2=subb(t2,P2,bo); t3=subb(t3,P3,bo); }
    r[0]=t0; r[1]=t1; r[2]=t2; r[3]=t3;
}

inline void mod_sqr(thread ulong r[4], thread const ulong a[4]) { mod_mul(r, a, a); }

// === Modular inverse (addition chain, p-2) ===
inline void mod_inv(thread ulong r[4], thread const ulong a[4]) {
    ulong x2[4],x3[4],x6[4],x9[4],x11[4],x22[4],x44[4],x88[4],x176[4],x220[4],x223[4],t[4];
    mod_sqr(x2,a); mod_mul(x2,x2,a);
    mod_sqr(x3,x2); mod_mul(x3,x3,a);
    mod_sqr(x6,x3); mod_sqr(x6,x6); mod_sqr(x6,x6); mod_mul(x6,x6,x3);
    mod_sqr(x9,x6); mod_sqr(x9,x9); mod_sqr(x9,x9); mod_mul(x9,x9,x3);
    mod_sqr(x11,x9); mod_sqr(x11,x11); mod_mul(x11,x11,x2);
    for(int i=0;i<4;i++)x22[i]=x11[i]; for(int i=0;i<11;i++)mod_sqr(x22,x22); mod_mul(x22,x22,x11);
    for(int i=0;i<4;i++)x44[i]=x22[i]; for(int i=0;i<22;i++)mod_sqr(x44,x44); mod_mul(x44,x44,x22);
    for(int i=0;i<4;i++)x88[i]=x44[i]; for(int i=0;i<44;i++)mod_sqr(x88,x88); mod_mul(x88,x88,x44);
    for(int i=0;i<4;i++)x176[i]=x88[i]; for(int i=0;i<88;i++)mod_sqr(x176,x176); mod_mul(x176,x176,x88);
    for(int i=0;i<4;i++)x220[i]=x176[i]; for(int i=0;i<44;i++)mod_sqr(x220,x220); mod_mul(x220,x220,x44);
    mod_sqr(x223,x220); mod_sqr(x223,x223); mod_sqr(x223,x223); mod_mul(x223,x223,x3);
    for(int i=0;i<4;i++)t[i]=x223[i]; for(int i=0;i<23;i++)mod_sqr(t,t); mod_mul(t,t,x22);
    for(int i=0;i<5;i++)mod_sqr(t,t); mod_mul(t,t,a);
    for(int i=0;i<3;i++)mod_sqr(t,t); mod_mul(t,t,x2);
    mod_sqr(t,t); mod_sqr(t,t); mod_mul(t,t,a);
    r[0]=t[0]; r[1]=t[1]; r[2]=t[2]; r[3]=t[3];
}

// === Point operations ===
inline bool limbs_zero(thread const ulong a[4]) { return (a[0]==0)&(a[1]==0)&(a[2]==0)&(a[3]==0); }

inline void jac_double(thread ulong rx[4], thread ulong ry[4], thread ulong rz[4],
                       thread const ulong px[4], thread const ulong py[4], thread const ulong pz[4]) {
    if (limbs_zero(pz)||limbs_zero(py)) { for(int i=0;i<4;i++){rx[i]=0;ry[i]=0;rz[i]=0;} return; }
    ulong y2[4],s[4],m[4],t[4],y4[4];
    mod_sqr(y2,py); mod_mul(s,px,y2); mod_add(s,s,s); mod_add(s,s,s);
    mod_sqr(m,px); mod_add(t,m,m); mod_add(m,t,m);
    mod_sqr(rx,m); mod_sub(rx,rx,s); mod_sub(rx,rx,s);
    mod_sub(t,s,rx); mod_mul(t,m,t);
    mod_sqr(y4,y2); mod_add(y4,y4,y4); mod_add(y4,y4,y4); mod_add(y4,y4,y4);
    mod_sub(ry,t,y4);
    mod_mul(rz,py,pz); mod_add(rz,rz,rz);
}

inline void jac_add_mixed(thread ulong rx[4], thread ulong ry[4], thread ulong rz[4],
                          thread const ulong px[4], thread const ulong py[4], thread const ulong pz[4],
                          thread const ulong qx[4], thread const ulong qy[4]) {
    if (limbs_zero(qx)&&limbs_zero(qy)) { for(int i=0;i<4;i++){rx[i]=px[i];ry[i]=py[i];rz[i]=pz[i];} return; }
    if (limbs_zero(pz)) { for(int i=0;i<4;i++){rx[i]=qx[i];ry[i]=qy[i];} rz[0]=1;rz[1]=0;rz[2]=0;rz[3]=0; return; }
    ulong z2[4],z3[4],u2[4],s2[4],h[4],rr[4],h2[4],h3[4],v[4],tmp[4];
    mod_sqr(z2,pz); mod_mul(z3,z2,pz); mod_mul(u2,qx,z2); mod_mul(s2,qy,z3);
    mod_sub(h,u2,px); mod_sub(rr,s2,py);
    if (limbs_zero(h)) {
        if (limbs_zero(rr)) { jac_double(rx,ry,rz,px,py,pz); }
        else { for(int i=0;i<4;i++){rx[i]=0;ry[i]=0;rz[i]=0;} }
        return;
    }
    mod_sqr(h2,h); mod_mul(h3,h2,h); mod_mul(v,px,h2);
    mod_sqr(tmp,rr); mod_sub(tmp,tmp,h3); mod_sub(tmp,tmp,v); mod_sub(rx,tmp,v);
    mod_sub(tmp,v,rx); mod_mul(tmp,rr,tmp);
    ulong yh3[4]; mod_mul(yh3,py,h3); mod_sub(ry,tmp,yh3);
    mod_mul(rz,pz,h);
}

// === Windowed scalar*G ===
inline void scalar_mul_g(thread ulong rx[4], thread ulong ry[4], thread ulong rz[4],
                         thread const ulong scalar[4], device const ulong* g_table) {
    for(int i=0;i<4;i++){rx[i]=0;ry[i]=0;rz[i]=0;}
    for (uint w = 0; w < 32u; ++w) {
        uint limb_idx = w >> 3;
        uint byte_in_limb = w & 7u;
        uint byte_val = (uint)((scalar[limb_idx] >> (8u * byte_in_limb)) & 0xFFu);
        if (byte_val == 0u) continue;
        uint base = (w * 256u + byte_val) * 8u;
        ulong qx[4] = {g_table[base],g_table[base+1],g_table[base+2],g_table[base+3]};
        ulong qy[4] = {g_table[base+4],g_table[base+5],g_table[base+6],g_table[base+7]};
        ulong nx[4],ny[4],nz[4];
        jac_add_mixed(nx,ny,nz,rx,ry,rz,qx,qy);
        for(int i=0;i<4;i++){rx[i]=nx[i];ry[i]=ny[i];rz[i]=nz[i];}
    }
}

// === SHA-256 (33 bytes) ===
constant uint K256[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

inline void sha256_33(thread const uchar pk[33], thread uchar hash[32]) {
    uint W[64];
    for(int i=0;i<8;i++) W[i]=((uint)pk[i*4]<<24)|((uint)pk[i*4+1]<<16)|((uint)pk[i*4+2]<<8)|(uint)pk[i*4+3];
    W[8]=((uint)pk[32]<<24)|(0x80u<<16);
    W[9]=0;W[10]=0;W[11]=0;W[12]=0;W[13]=0;W[14]=0;W[15]=264;
    for(int i=16;i<64;i++){
        uint s0=((W[i-15]>>7)|(W[i-15]<<25))^((W[i-15]>>18)|(W[i-15]<<14))^(W[i-15]>>3);
        uint s1=((W[i-2]>>17)|(W[i-2]<<15))^((W[i-2]>>19)|(W[i-2]<<13))^(W[i-2]>>10);
        W[i]=W[i-16]+s0+W[i-7]+s1;
    }
    uint a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d=0xa54ff53au;
    uint e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u;
    for(int i=0;i<64;i++){
        uint S1=((e>>6)|(e<<26))^((e>>11)|(e<<21))^((e>>25)|(e<<7));
        uint ch=(e&f)^(~e&g);
        uint t1=h+S1+ch+K256[i]+W[i];
        uint S0=((a>>2)|(a<<30))^((a>>13)|(a<<19))^((a>>22)|(a<<10));
        uint maj=(a&b)^(a&c)^(b&c);
        uint t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    a+=0x6a09e667u;b+=0xbb67ae85u;c+=0x3c6ef372u;d+=0xa54ff53au;
    e+=0x510e527fu;f+=0x9b05688cu;g+=0x1f83d9abu;h+=0x5be0cd19u;
    uint H[8]={a,b,c,d,e,f,g,h};
    for(int i=0;i<8;i++){hash[i*4]=(uchar)(H[i]>>24);hash[i*4+1]=(uchar)(H[i]>>16);hash[i*4+2]=(uchar)(H[i]>>8);hash[i*4+3]=(uchar)H[i];}
}

// === RIPEMD-160 (32 bytes) ===
constant uint RKL[5]={0x00000000u,0x5A827999u,0x6ED9EBA1u,0x8F1BBCDCu,0xA953FD4Eu};
constant uint RKR[5]={0x50A28BE6u,0x5C4DD124u,0x6D703EF3u,0x7A6D76E9u,0x00000000u};
constant int RRL[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
constant int RRR[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
constant int RSL[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
constant int RSR[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};

inline uint rrotl(uint x, int n) { return (x<<n)|(x>>(32-n)); }

inline void ripemd160_32(thread const uchar sha[32], thread uchar h160[20]) {
    uint X[16];
    for(int i=0;i<8;i++) X[i]=(uint)sha[i*4]|((uint)sha[i*4+1]<<8)|((uint)sha[i*4+2]<<16)|((uint)sha[i*4+3]<<24);
    X[8]=0x80u; for(int i=9;i<14;i++)X[i]=0; X[14]=256u; X[15]=0;

    uint AL=0x67452301u,BL=0xEFCDAB89u,CL=0x98BADCFEu,DL=0x10325476u,EL=0xC3D2E1F0u;
    uint AR=AL,BR=BL,CR=CL,DR=DL,ER=EL;

    for(int j=0;j<16;j++){
        uint tL=rrotl(AL+(BL^CL^DL)+X[RRL[j]]+RKL[0],RSL[j])+EL; AL=EL;EL=DL;DL=rrotl(CL,10);CL=BL;BL=tL;
        uint tR=rrotl(AR+(BR^(CR|~DR))+X[RRR[j]]+RKR[0],RSR[j])+ER; AR=ER;ER=DR;DR=rrotl(CR,10);CR=BR;BR=tR;
    }
    for(int j=16;j<32;j++){
        uint tL=rrotl(AL+((BL&CL)|(~BL&DL))+X[RRL[j]]+RKL[1],RSL[j])+EL; AL=EL;EL=DL;DL=rrotl(CL,10);CL=BL;BL=tL;
        uint tR=rrotl(AR+((BR&DR)|(CR&~DR))+X[RRR[j]]+RKR[1],RSR[j])+ER; AR=ER;ER=DR;DR=rrotl(CR,10);CR=BR;BR=tR;
    }
    for(int j=32;j<48;j++){
        uint tL=rrotl(AL+((BL|~CL)^DL)+X[RRL[j]]+RKL[2],RSL[j])+EL; AL=EL;EL=DL;DL=rrotl(CL,10);CL=BL;BL=tL;
        uint tR=rrotl(AR+((BR|~CR)^DR)+X[RRR[j]]+RKR[2],RSR[j])+ER; AR=ER;ER=DR;DR=rrotl(CR,10);CR=BR;BR=tR;
    }
    for(int j=48;j<64;j++){
        uint tL=rrotl(AL+((BL&DL)|(CL&~DL))+X[RRL[j]]+RKL[3],RSL[j])+EL; AL=EL;EL=DL;DL=rrotl(CL,10);CL=BL;BL=tL;
        uint tR=rrotl(AR+((BR&CR)|(~BR&DR))+X[RRR[j]]+RKR[3],RSR[j])+ER; AR=ER;ER=DR;DR=rrotl(CR,10);CR=BR;BR=tR;
    }
    for(int j=64;j<80;j++){
        uint tL=rrotl(AL+(BL^(CL|~DL))+X[RRL[j]]+RKL[4],RSL[j])+EL; AL=EL;EL=DL;DL=rrotl(CL,10);CL=BL;BL=tL;
        uint tR=rrotl(AR+(BR^CR^DR)+X[RRR[j]]+RKR[4],RSR[j])+ER; AR=ER;ER=DR;DR=rrotl(CR,10);CR=BR;BR=tR;
    }

    uint H0=0x67452301u,H1=0xEFCDAB89u,H2=0x98BADCFEu,H3=0x10325476u,H4=0xC3D2E1F0u;
    uint t=H1+CL+DR; H1=H2+DL+ER; H2=H3+EL+AR; H3=H4+AL+BR; H4=H0+BL+CR; H0=t;
    uint Hf[5]={H0,H1,H2,H3,H4};
    for(int i=0;i<5;i++){h160[i*4]=(uchar)Hf[i];h160[i*4+1]=(uchar)(Hf[i]>>8);h160[i*4+2]=(uchar)(Hf[i]>>16);h160[i*4+3]=(uchar)(Hf[i]>>24);}
}

// === Main kernel: symmetric group addition ===
// Each thread owns one GROUP of GROUP_SIZE = 2*GROUP_H+1 consecutive keys,
// centered on scalar C:
//   1. C = center_index * G  (one windowed scalar_mul_g + one mod_inv to affinize)
//   2. For i=1..H, the affine multiples i*G are precomputed constants (ig_table).
//      The points C +/- i*G share the SAME x-difference den[i] = x(iG) - Cx,
//      so one batch inversion over the H denominators yields every slope.
//   3. Each of the 2H output points costs ~3 field muls and is produced already
//      in affine form (no separate per-point inverse). Hash + compare on the fly.
// This replaces the 2H-long Jacobian incremental-add chain (the old bottleneck)
// with H cheap affine combinations sharing a single mod_inv.
// Mirrors the CPU backend's verified search_incremental (backend_cpu.h).

#define GROUP_H 256
#define GROUP_SIZE (2 * GROUP_H + 1)

inline void affine_to_h160(thread const ulong ax[4], thread const ulong ay[4],
                           thread uchar h160[20]) {
    uchar pubkey[33];
    pubkey[0] = (uchar)(0x02u | (uint)(ay[0] & 1ul));
    for (int i = 0; i < 4; ++i) {
        ulong l = ax[3-i];
        for (int j = 0; j < 8; ++j) pubkey[1+i*8+j] = (uchar)(l >> (56-8*j));
    }
    uchar sha[32];
    sha256_33(pubkey, sha);
    ripemd160_32(sha, h160);
}

// Compare h160(ax,ay) to target; on match, CAS-publish private key = start+idx.
inline void check_point(thread const ulong ax[4], thread const ulong ay[4],
                        device const uchar* target_h160, ulong start_lo, ulong start_hi,
                        ulong key_idx, device ulong* match_lo, device ulong* match_hi,
                        device atomic_uint* match_found) {
    uchar h160[20];
    affine_to_h160(ax, ay, h160);
    for (int b = 0; b < 20; ++b) if (h160[b] != target_h160[b]) return;
    uint expected = 0u;
    if (atomic_compare_exchange_weak_explicit(match_found, &expected, 1u,
            memory_order_relaxed, memory_order_relaxed)) {
        ulong c = 0;
        ulong klo = addc(start_lo, key_idx, c);
        ulong khi = addc(start_hi, 0, c);
        match_lo[0] = klo;
        match_hi[0] = khi;
    }
}

kernel void puzzle_search(
    device const ulong*       g_table      [[buffer(0)]],
    device const uchar*       target_h160  [[buffer(1)]],
    constant     ulong&       start_lo     [[buffer(2)]],
    constant     ulong&       start_hi     [[buffer(3)]],
    constant     ulong&       total_keys   [[buffer(4)]],
    device       ulong*       match_lo     [[buffer(5)]],
    device       ulong*       match_hi     [[buffer(6)]],
    device       atomic_uint* match_found  [[buffer(7)]],
    device const ulong*       ig_table     [[buffer(8)]],
    uint                      gid          [[thread_position_in_grid]])
{
    if (atomic_load_explicit(match_found, memory_order_relaxed) != 0u) return;

    // This thread owns the group of GROUP_SIZE keys starting at g_start.
    ulong g_start = (ulong)gid * (ulong)GROUP_SIZE;
    if (g_start >= total_keys) return;

    // Center scalar index (relative to start) = g_start + GROUP_H.
    ulong center_index = g_start + (ulong)GROUP_H;
    ulong kc[4];
    { ulong c = 0;
      kc[0] = addc(start_lo, center_index, c);
      kc[1] = addc(start_hi, 0, c);
      kc[2] = c; kc[3] = 0; }

    // C = center_index * G, affinized.
    ulong CXj[4], CYj[4], CZj[4];
    scalar_mul_g(CXj, CYj, CZj, kc, g_table);
    if (limbs_zero(CZj)) return;             // center at infinity (impossible for real range)
    ulong Cx[4], Cy[4];
    { ulong zi[4], zi2[4], zi3[4];
      mod_inv(zi, CZj); mod_sqr(zi2, zi); mod_mul(zi3, zi2, zi);
      mod_mul(Cx, CXj, zi2); mod_mul(Cy, CYj, zi3); }

    // Denominators den[i] = x(i*G) - Cx, i=1..H.  ig_table slot i = (x[4],y[4]).
    ulong den[GROUP_H + 1][4];
    bool degenerate = false;
    for (uint i = 1; i <= GROUP_H; ++i) {
        ulong igx[4] = {ig_table[i*8], ig_table[i*8+1], ig_table[i*8+2], ig_table[i*8+3]};
        mod_sub(den[i], igx, Cx);
        if (limbs_zero(den[i])) degenerate = true;
    }

    if (degenerate) {
        // Astronomically rare (center.x collides with a table x). Recompute the
        // whole group directly so we never skip or mis-emit a key.
        for (uint m = 0; m < GROUP_SIZE; ++m) {
            ulong idx = g_start + (ulong)m;
            if (idx >= total_keys) break;
            ulong k[4]; { ulong c = 0;
              k[0] = addc(start_lo, idx, c); k[1] = addc(start_hi, 0, c); k[2]=c; k[3]=0; }
            ulong px[4], py[4], pz[4];
            scalar_mul_g(px, py, pz, k, g_table);
            if (limbs_zero(pz)) continue;
            ulong zi[4], zi2[4], zi3[4], ax[4], ay[4];
            mod_inv(zi, pz); mod_sqr(zi2, zi); mod_mul(zi3, zi2, zi);
            mod_mul(ax, px, zi2); mod_mul(ay, py, zi3);
            check_point(ax, ay, target_h160, start_lo, start_hi, idx,
                        match_lo, match_hi, match_found);
        }
        return;
    }

    // Batch invert den[1..H] (Montgomery trick): one mod_inv for the group.
    ulong pre[GROUP_H + 1][4];
    pre[1][0]=den[1][0]; pre[1][1]=den[1][1]; pre[1][2]=den[1][2]; pre[1][3]=den[1][3];
    for (uint i = 2; i <= GROUP_H; ++i) mod_mul(pre[i], pre[i-1], den[i]);
    ulong acc[4];
    mod_inv(acc, pre[GROUP_H]);
    ulong inv[GROUP_H + 1][4];
    for (int i = GROUP_H; i >= 1; --i) {
        if (i > 1) mod_mul(inv[i], acc, pre[i-1]);
        else { inv[1][0]=acc[0]; inv[1][1]=acc[1]; inv[1][2]=acc[2]; inv[1][3]=acc[3]; }
        mod_mul(acc, acc, den[i]);
    }

    // Center key (index g_start + GROUP_H).
    if (center_index < total_keys)
        check_point(Cx, Cy, target_h160, start_lo, start_hi, center_index,
                    match_lo, match_hi, match_found);

    // C +/- i*G for i=1..H, sharing inv[i]. -P just negates y.
    for (uint i = 1; i <= GROUP_H; ++i) {
        ulong igx[4] = {ig_table[i*8], ig_table[i*8+1], ig_table[i*8+2], ig_table[i*8+3]};
        ulong igy[4] = {ig_table[i*8+4], ig_table[i*8+5], ig_table[i*8+6], ig_table[i*8+7]};

        // C + i*G  -> index (GROUP_H + i)
        ulong up = g_start + (ulong)GROUP_H + (ulong)i;
        if (up < total_keys) {
            ulong dy[4], s[4], x3[4], y3[4], t[4];
            mod_sub(dy, igy, Cy);
            mod_mul(s, dy, inv[i]);
            mod_sqr(x3, s); mod_sub(x3, x3, Cx); mod_sub(x3, x3, igx);
            mod_sub(t, Cx, x3); mod_mul(y3, s, t); mod_sub(y3, y3, Cy);
            check_point(x3, y3, target_h160, start_lo, start_hi, up,
                        match_lo, match_hi, match_found);
        }

        // C - i*G  -> index (GROUP_H - i), uses Q = (igx, -igy)
        ulong ny[4];   // -igy mod p  (igy != 0 for i>=1)
        { ulong bo=0; ny[0]=subb(P0, igy[0], bo); ny[1]=subb(P1, igy[1], bo);
          ny[2]=subb(P2, igy[2], bo); ny[3]=subb(P3, igy[3], bo); }
        ulong down = g_start + (ulong)GROUP_H - (ulong)i;   // GROUP_H-i >= 0
        if (down < total_keys) {
            ulong dy2[4], s2[4], x3b[4], y3b[4], t2[4];
            mod_sub(dy2, ny, Cy);
            mod_mul(s2, dy2, inv[i]);
            mod_sqr(x3b, s2); mod_sub(x3b, x3b, Cx); mod_sub(x3b, x3b, igx);
            mod_sub(t2, Cx, x3b); mod_mul(y3b, s2, t2); mod_sub(y3b, y3b, Cy);
            check_point(x3b, y3b, target_h160, start_lo, start_hi, down,
                        match_lo, match_hi, match_found);
        }
    }
}


