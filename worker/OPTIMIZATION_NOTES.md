# Metal 后端性能优化记录

分支 `opt-metal-perf`。设备：Apple M3 Pro。性能均为**实测**（连真实协调者跑任务的 per-task MK/s），非 init benchmark。

## 已落地（已验证 9/9 + 已提交）

| 阶段 | 真实 MK/s | 加速 | commit |
|------|-----------|------|--------|
| baseline (clean main) | 14.5 | 1.0x | 6bedb7a |
| 增量点加 + Montgomery 批量求逆 (S=8) | 37 | 2.5x | 429e93c |
| 调 KEYS_PER_THREAD → 128 | 83 | 5.7x | 22ea537 |
| **对称群加 C ± i·G（移植 CPU 方案）** | **158** | **10.9x** | symgroup |
| **调 GROUP_H → 256** | **180** | **12.4x** | symgroup |

**核心改动（最终版，对称群加）**：每个 GPU 线程负责一个 group（`2H+1` 个连续 key，H=256），中心 C：
- `C = center_index·G` 算一次并仿射化（1 次 mod_inv）
- 预计算 `i·G` 仿射表（host 端 `b_igtable`），`C±i·G` 共享同一分母 `den[i]=x(iG)−Cx`
- H 个分母一次批量求逆（第 2 次 mod_inv），产出 `2H` 个点，每点 ~3 乘且**直接是仿射坐标**
- degenerate（den[i]=0）整组回退逐 key `scalar_mul_g + 单点求逆`

替换了之前 2H 长的 Jacobian 增量加链（旧瓶颈）。GPU 关键适配：**每线程独立负责一组、组内自己串行求逆、不跨线程协作**——恰好避开 CUDA 那次失败的坑（`62f8211` 回退，根因是压给 tid==0 + `__syncthreads()`）。

`GROUP_H` 扫描（H vs MK/s）：64→158, 128→177, **256→180**, 512→124（越过拐点，~24KB/线程栈杀 occupancy）。256 最优。

> **CPU 经验完全适用**：文档框里 CPU 给的三条（对称 ±i·G 共享分母、degenerate 整组 fallback、批量求逆）在 GPU 上原样移植即生效，实测 1.9x（超出预估 1.4-1.9x 上限）。degenerate 路径用 `test_degen`（k=1..300 跨多组穷举）单独验证 300/300。
> 注：跨组推进中心点（CPU 的经验 C）在 GPU 版**不需要**——每线程只处理一个 group，比 CPU 更简单，也不会有"错误沿增量链传播"的风险。

## 早期已落地（被对称群加取代，保留历史）

- 增量点加 + Montgomery 批量求逆（S=128，83 MK/s）：每线程 S 个连续 key，Jacobian mixed-add 递推 + 共享一次 inv。对称群加是其升级版。

## 已验证为负优化 / 无效（不要再试）

- **dedicated mod_sqr**（对称项算一次加两次）：83 → **61 MK/s，负优化**。原因：进位传播需要分支/可变循环，伤 GPU 直线流水。通用 `mod_mul(a,a)` 的无分支固定结构反而更快。注意：`opt1-modsqr` 分支上还有一份**有 bug** 的手写 mod_sqr（129-bit doubling 移位错误，verify 失败），更不要用。
- **threadgroup 宽度** 32/64/128/256：全部 ~81-83 MK/s，无差异。GPU 受 register/occupancy 限制，不是 threadgroup 调度限制。

> **📌 CPU 后端印证**：CPU 的 `mod_sqr` 也是直接调 `mod_mul(a,a)`、没单独实现，和这里"dedicated mod_sqr 是负优化"的结论方向一致。CPU 上理论上有分支预测、单独写 mod_sqr 或许有小收益，但 GPU 这条 83→61 的实测数据说明收益不稳，不值得为它冒结构复杂化的险。

## 待尝试（未实现，预期有效，风险高）

### 1. ✅ 已完成 — 仿射对称群加 + 批量求逆（实际拿到 1.9x → 180 MK/s）
见上方"已落地"。下面保留原始分析与 CPU 经验框供参考。
当前瓶颈是 127 次 Jacobian `jac_add_mixed`（每次 ~11乘+3平方 ≈ 1780 域运算/线程，占大头）。

思路：每线程算 `start + i·G`，i=0..S-1。其中：
- `start = k_base·G`，用现有 scalar_mul_g 算一次并仿射化
- `i·G` 是 **host 端预计算的仿射常量表**（i=0..S-1），新开一个 MTLBuffer 上传
- 所有分母 `d_i = x(iG) − x(start)` 可**独立预算**（无串行依赖）→ 一次批量求逆得全部 1/d_i
- 每点加法只需 λ=(y(iG)−y(start))·(1/d_i)，x=λ²−x(start)−x(iG)，y=λ(x(start)−x)−y(start) ≈ 2乘+1平方

**正确性陷阱（必须处理）**：当 k_base = ±i 时 d_i=0（doubling/无穷远情形）。小 key 测试（puzzle1 k=1）必然触发。这些点必须走 Jacobian fallback，否则漏目标 key。9/9 区间测试能抓到这个 bug。

实现：host 端加 `build_step_table(S)` 生成 i·G 仿射表 + 新 buffer；kernel 重写主循环。

> **📌 来自 CPU 后端的实战经验（这个方案 CPU 已落地并验证，可直接参考）**
>
> CPU 后端（`backend_cpu.h` 的 `search_incremental`）已经把这个方案做完了，单线程 +118%、整体 4.5×。两点经验能直接帮到 Metal：
>
> **(A) 用对称 `C ± i·G`，不要单向 `start + i·G` —— 求逆再省一半。**
> 关键观察：`+i·G` 和 `−i·G` 的 x 坐标相同（`−P` 只翻转 y），所以 `d_i = x(C) − x(iG)` 对 `+i` 和 `−i` **是同一个分母**。
> 做法：以 `C = (start + H)·G` 为中心（H = 半宽），一组 `2H+1` 个 key。批量求逆只需对 `i=1..H` 的 H 个分母做（不是 2H 个），却能产出 `2H` 个点：
> - `C + i·G`：λ=(y(iG)−Cy)·(1/d_i)
> - `C − i·G`：用 `(x(iG), −y(iG))`，复用**同一个 `1/d_i`**，λ=(−y(iG)−Cy)·(1/d_i)
>
> 即每个输出点 ~3 乘、且**直接是仿射坐标**（省掉单独 affinize 那次 1 inv + 3 mul）。这比上面单向方案再省一倍求逆规模，实际收益可能高于估的 1.4–1.9×。
>
> **(B) 退化 fallback 的具体写法（你点对了陷阱，这是验证过的解法）。**
> CPU 版的处理：算完所有分母后，把它们 OR 在一起判零（`(den[i][0]|den[i][1]|den[i][2]|den[i][3])==0` 命中即 degenerate）。一旦命中，**整组**回退到逐 key 直接 `scalar_mul_g + 单点求逆`（不是只修那一个 key），算完后强制下一组重新算中心点。代价：极罕见时损失一组的速度，但绝不漏 / 不误报。9/9 区间测试（尤其 puzzle1 k=1）能抓到漏掉这步的 bug。
>
> **(C) 跨组推进中心点也复用同一次批量求逆。**
> 把"中心点前进一步" `C += (2H+1)·G` 的那个分母 `d_step = x(C) − x((2H+1)·G)` 也塞进同一批求逆里（CPU 版放在 `den[0]`），这样组与组之间推进中心**不额外花求逆**。注意：中心是增量推进的，一旦某组 degenerate 必须把中心标记失效、下组用 scalar_mul 重算，否则错误会沿增量链传播污染后续所有组。
>
> CPU 实现 + 穷举覆盖测试见 `backend_cpu.h` 和 `test_coverage.cpp`（每个偏移全扫，非抽样）。

### 2. co-Z 加法（Meloni）替换 Jacobian mixed-add（中优先级）
连续点共享 Z，增量步 ~5乘+2平方（vs 11乘+3平方）。比方案 1 简单但收益略低。与方案 1 二选一。

### 3. 低风险小优化（收益小但几乎无风险）
- scalar_mul_g 内层循环展开 / 预取
- SHA256 / RIPEMD160 常量表用 constant 地址空间确认已最优
- 减少 affine 转换中间变量的寄存器占用

## 安全网
`worker/test_metal_correctness.cpp`：区间搜索测试，把已知 key 放在 batch 内非零偏移（1, 37, 1000, 65535, 1M）处，要求返回精确私钥 + 无误报。任何 kernel 改动后必跑，干净 main 为 golden 9/9。

构建测试：
```
cd worker && make
clang++ -std=c++17 -O3 -march=native -I. -Ikernels -fobjc-arc \
  test_metal_correctness.cpp metal_solver.o \
  -framework Metal -framework Foundation -framework CoreGraphics -lpthread \
  -o test_metal_correctness && ./test_metal_correctness
```

实测性能：`./puzzle_worker --url http://<coordinator>:8080 --backend metal`，看 per-task `MK/s`（不是开头的 init `Speed:`，那个不准）。

---

## 附：CPU 后端优化总览（供 Metal 借鉴，算法层面硬件无关）

CPU 后端（x86, AMD EPYC Zen2）已落地四项，单线程 1.11 → 4.96 MK/s（4.5×），8 核实测 8.8 → 39 MK/s。其中**对称点群加是硬件无关的算法优化，与上面 Metal 待办 #1 同源**：

| 优化 | 类型 | 单线程效果 | 可移植到 GPU? |
|------|------|-----------|--------------|
| SHA-256 → SHA-NI 硬件指令 | x86 专属 | +34% | ❌（GPU 哈希非瓶颈，SIMT 已天然并行）|
| mod_mul → MULX/ADCX/ADOX 双进位链 | x86 专属 | +7% | ❌（GPU 无此指令）|
| RIPEMD-160 → AVX2 8 路 SIMD | x86 专属 | +43% | ❌（同上，SIMT 已并行）|
| **对称点群加 + 批量共享求逆** | **纯算法** | **+118%** | ✅ **= Metal 待办 #1，见上方框** |

历史教训：`9e245bf` 曾在 **CUDA** 上试过批量求逆但被回退（`62f8211`），根因是把 prefix product / back-propagation **全压给 tid==0 单线程串行 + 全块 `__syncthreads()`**，occupancy 崩盘。Metal 现在"每线程独立负责一组 key、组内自己串行求逆、不跨线程协作"的做法**恰好避开了这个坑**——这是正确姿势，别改成跨线程协作版。

代码参考：`backend_cpu.h`（`search_incremental` = 对称群加 + 退化 fallback + 跨组推进），`test_coverage.cpp`（穷举每个偏移的覆盖测试）。

---

## Mac CPU (Apple Silicon / ARM) 优化

针对 M3 Pro（6 性能核 + 6 能效核）的专项优化。**关键结论：x86 的三个优化（SHA-NI / AVX2 / MULX-ADX）是指令集专属内联汇编/intrinsics，在 ARM 上条件编译全部 fallback 到标量，无法直接复用——必须重写 ARM 等价版。** 实测（真实 worker，全核）约 18 → 46 MK/s（~2.5x）。

| 优化 | 类型 | 效果 | commit |
|------|------|------|--------|
| work-stealing 负载均衡 | 调度（异构核） | 修正能效核拖尾 | opt-cpu-arm |
| ARMv8 SHA-256 硬件指令 | ARM 专属 | 单核 +60% | opt-cpu-arm |
| NEON 4 路 RIPEMD-160 | ARM 专属 | 哈希 2.53x，整体再 +38% | opt-cpu-arm |

吞吐（微基准 real backend_cpu，高位 key）：

| 线程 | baseline | +work-steal | +ARM SHA | +NEON RMD |
|------|----------|-------------|----------|-----------|
| 1 | 2.31 | 2.31 | 3.69 | **5.90** |
| 6 | 12.85 | 13.08 | 19.92 | **30.79** |
| 12 | 18.30 | 19.07 | 31.09 | **42.90** |

### 关键经验（踩过的坑）

> **(A) `-march=native` 在 Apple clang 的 arm64 上 _不_ 启用 crypto 特性宏。**
> 这是最隐蔽的坑：`__ARM_FEATURE_SHA2` / `__ARM_NEON` 在裸 clang 默认 target 下有，但加了 `-march=native` 反而消失，导致 ARM SHA 路径永远走不到。必须用 **`-mcpu=native`**（保留全部 native 调优 + 启用 crypto）。Makefile 已按 `uname -m == arm64` 切换。验证：`echo | clang++ -mcpu=native -dM -E -x c++ - | grep ARM_FEATURE_SHA2`。

> **(B) 异构 P/E 核：静态均分是错的。**
> 原 `backend_cpu` 把 chunk 平均切 N 份，总耗时被最慢的能效核拖住（性能核算完空等）。改为原子计数器分发 GROUP_SIZE 对齐的小 tile（work-stealing），性能核自然多抢。扩展比从 ~6 线程后塌陷恢复到 12 线程 ~8x。

> **(C) ARM SHA-256 结构比 x86 SHA-NI 干净。**
> 无需 x86 那套 state shuffle（abef/cdgh 重排）。直接 `vsha256hq_u32` + `vsha256h2q_u32` 配对，`vsha256su0/su1` 做消息扩展，load/store 用 `vrev32q_u8` 做大端字节序转换。

> **(D) NEON 只有 128-bit → 4 路（x86 AVX2 是 256-bit → 8 路）。**
> RIPEMD 变量循环移位用 `vshlq_u32(x, vdupq_n_s32(n))` + `vshlq_u32(x, vdupq_n_s32(n-32))` 组合（NEON 负移位量 = 右移）。对称群加每组产出一批点，攒 4 个一组喂给 `pubkey_to_hash160_4way`。

代码参考：`kernels/hash.h`（`sha256_33bytes_arm` / `ripemd160_4way`，分别 gated on `__ARM_FEATURE_SHA2` / `__ARM_NEON`），`backend_cpu.h::search`（work-stealing）。验证：ARM SHA 0/100k mismatch、NEON RMD 0/50k mismatch、`make test` 9/9 + coverage 1872/0。
