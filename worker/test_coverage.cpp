#include "backend_cpu.h"
#include <cstdio>
#include <cstring>
static uint64_t* g;
static void h160_of(uint64_t key,uint8_t*target){
    uint64_t k[4]={key,0,0,0};
    secp256k1::JacobianPoint P; secp256k1::scalar_mul_g_windowed(P,k,g);
    uint64_t zi[4],zi2[4],zi3[4],ax[4],ay[4];
    secp256k1::mod_inv(zi,P.Z); secp256k1::mod_sqr(zi2,zi); secp256k1::mod_mul(zi3,zi2,zi);
    secp256k1::mod_mul(ax,P.X,zi2); secp256k1::mod_mul(ay,P.Y,zi3);
    uint8_t pk[33]; pk[0]=0x02|(uint8_t)(ay[0]&1);
    for(int li=0;li<4;li++){uint64_t l=ax[3-li];for(int j=0;j<8;j++)pk[1+li*8+j]=(uint8_t)(l>>(56-8*j));}
    hash::pubkey_to_hash160(pk,target);
}
int main(){
    g=new uint64_t[secp256k1::G_TABLE_ULONGS]; secp256k1::build_g_table(g);
    CPUBackend be(1); be.init();
    int fails=0, checked=0;
    // Search a fixed window [start, start+size). For each target key position p in
    // [0,size), the key must be found exactly. Cover >2 full groups + boundaries.
    uint64_t start=5000000;
    uint64_t size=1300;   // > 2*GROUP_SIZE(513)=1026, exercises 3 groups + partial
    for(uint64_t p=0;p<size;p++){
        uint64_t key=start+p;
        uint8_t target[20]; h160_of(key,target);
        uint64_t fl=0,fh=0;
        bool ok=be.search(start,0,size,target,fl,fh);
        checked++;
        if(!(ok&&fl==key&&fh==0)){ fails++; if(fails<=10) printf("MISS p=%llu key=%llu ok=%d got=%llu\n",
            (unsigned long long)p,(unsigned long long)key,ok,(unsigned long long)fl);}
    }
    // Multi-thread coverage: same but 8 threads (thread-boundary slicing)
    CPUBackend be8(8); be8.init();
    uint64_t size2=4000;
    for(uint64_t p=0;p<size2;p+=7){  // sample every 7 to keep runtime sane
        uint64_t key=start+p;
        uint8_t target[20]; h160_of(key,target);
        uint64_t fl=0,fh=0;
        bool ok=be8.search(start,0,size2,target,fl,fh);
        checked++;
        if(!(ok&&fl==key&&fh==0)){ fails++; if(fails<=20) printf("MT MISS p=%llu key=%llu ok=%d got=%llu\n",
            (unsigned long long)p,(unsigned long long)key,ok,(unsigned long long)fl);}
    }
    printf("checked=%d fails=%d %s\n",checked,fails,fails?"FAIL":"ALL PASS");
    return fails?1:0;
}
