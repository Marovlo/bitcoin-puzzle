# 性能优化记录

worker 在各后端的优化进度、方法与踩过的坑。**所有性能数字均为实测**——连真实协调者跑任务的 per-task `MK/s`，**不是启动时的 init benchmark**（init 从 key=1 开始、落在 degenerate 区，数字偏低不准）。

## 当前状态总览

| 后端 | baseline | 当前 | 加速 | 设备 |
|------|----------|------|------|------|
| Metal | 14.5 MK/s | **180 MK/s** | 12.4x | Apple M3 Pro |
| CPU (ARM) | ~18 MK/s | **46 MK/s** | 2.5x | M3 Pro 12 线程 |
| CPU (x86) | 8.8 MK/s | **39 MK/s** | 4.5x | AMD EPYC Zen2 8 核 |
| CUDA | 未优化 | 未优化 | — | 算法落后一代，见 §3 |

## 心法（先读这条）

性能分两层，**算法层 > 微架构层**：

1. **算法层（硬件无关，收益最大）**：减少每个 key 的椭圆曲线运算量。核心是**对称群加 + 批量共享求逆**——把最贵的 `mod_inv`（~270 乘）和 `scalar_mul_g`（~350 乘）摊薄到一整组 key 上。这套方法在 CPU/Metal 都拿到了数量级提升，**可移植到任何后端**（见 §1）。
2. **微架构层（设备专属）**：哈希用硬件指令（x86 SHA-NI/AVX2、ARM SHA2/NEON、CUDA PTX）、负载均衡、occupancy 调参。收益次之，且**不可跨指令集复用**——每种设备要单独写（见 §2）。

**优化新后端的正确顺序**：先移植 §1 的算法（最大收益、跨平台通用），再做 §2 的设备专属微调。别一上来就抠指令。

---

# §1 通用优化（算法层，硬件无关，所有后端适用）

## 1.1 对称群加 C ± i·G + 批量共享求逆 ★核心

**这是收益最大的优化，CPU/Metal 已验证，CUDA 待移植。**

**问题**：朴素实现 1 线程/key，每个 key 都做完整 `scalar_mul_g` + `mod_inv`，两者各占约一半开销。

**方法**：每个线程负责一个 **group**（`2H+1` 个连续 key），以中心 C 为锚：
- `C = center_index·G` 算一次并仿射化（1 次 `mod_inv`）
- `i·G` 仿射表（i=1..H）host 端预计算好传进 kernel
- 关键观察：`+i·G` 与 `−i·G` 的 **x 坐标相同**（`−P` 只翻转 y），所以分母 `den[i] = x(iG) − Cx` 对 `+i` 和 `−i` **是同一个**
- H 个分母**一次批量求逆**（Montgomery trick，第 2 次也是唯一一次额外 `mod_inv`）→ 产出 `2H` 个点，每点仅 ~3 乘，且**直接是仿射坐标**（省掉单独 affinize）
- 替换了旧的 2H 长 Jacobian 增量加链（旧瓶颈，每步 ~11乘+3平方）

**正确性陷阱（必须处理）**：当 `center_index = ±i` 时 `den[i]=0`（doubling/无穷远）。小 key（puzzle1 k=1）必然触发。处理：所有分母 OR 在一起判零，命中则**整组**回退逐 key `scalar_mul_g + 单点求逆`（不是只修那一个），算完强制下组重算中心。绝不漏 key / 不误报。

**实测收益**：CPU 单线程 +118%；Metal 1.9x（83→158 MK/s，超出预估上限）。

**参考实现**：`backend_cpu.h::search_incremental`（最干净、穷举验证过）；Metal 落地见 `kernels/puzzle.metal::puzzle_search` + `metal_solver.mm` 的 ig_table 构建。

## 1.2 群大小 H 调参

H 越大，求逆和 scalar_mul 摊得越薄；但每线程要存一组中间量，寄存器/栈压力上升，过大伤 occupancy。**典型 U 型曲线，需 sweep**：

- Metal `GROUP_H` 扫描：64→158, 128→177, **256→180**, 512→124（越过拐点，~24KB/线程栈杀 occupancy）。M3 Pro 最优 256。
- CPU 用 H=256。CUDA 寄存器更紧，预计要从 16/32/64 往上试。

## 1.3 GPU 落地的关键适配（CUDA 移植必读）

> **每线程独立负责一整组、组内自己串行求逆、线程间不协作。** 不要跨线程做 prefix-product / back-propagation。
>
> 历史教训：CUDA 上 `9e245bf` 曾试批量求逆，把 prefix/back-prop **全压给 `tid==0` + 全块 `__syncthreads()`**，occupancy 崩盘，被 `62f8211` 回退。Metal 现在"每线程一组、组内串行、零协作"的做法**恰好避开了这个坑**——这是正确姿势。代价是单线程寄存器压力大，用 H 调参平衡。
>
> 注：CPU 版有个"跨组增量推进中心点"的优化（`C += (2H+1)·G` 复用同批求逆），但 **GPU 版不需要**——每线程只处理一个 group，更简单，也没有"错误沿增量链传播"的风险。

---

# §2 设备专属优化（微架构层，不可跨指令集复用）

> **核心结论**：哈希（SHA-256 / RIPEMD-160）在 CPU 上是单核瓶颈，各平台都有硬件加速手段，但**全是指令集专属的 intrinsics/内联汇编，无法跨架构复用**——x86 的 SHA-NI/AVX2 在 ARM 上条件编译直接 fallback 到标量，必须为每种架构重写等价版。GPU 则因 SIMT 天然并行，哈希不是瓶颈，无需专门 SIMD 化。

## 2.1 Mac CPU (Apple Silicon / ARM) — 已落地 2.5x

设备 M3 Pro（6 性能核 + 6 能效核）。约 18 → 46 MK/s（全核实测）。

| 优化 | 类型 | 效果 |
|------|------|------|
| 动态自调度负载均衡 | 调度（平台无关，见下） | 修正异构核拖尾 |
| ARMv8 SHA-256 硬件指令 | ARM 专属 | 单核 +60% |
| NEON 4 路 RIPEMD-160 | ARM 专属 | 哈希 2.53x，整体再 +38% |

吞吐递进（微基准 real backend_cpu，高位 key）：

| 线程 | baseline | +自调度 | +ARM SHA | +NEON RMD |
|------|----------|---------|----------|-----------|
| 1 | 2.31 | 2.31 | 3.69 | **5.90** |
| 6 | 12.85 | 13.08 | 19.92 | **30.79** |
| 12 | 18.30 | 19.07 | 31.09 | **42.90** |

**踩过的坑：**

> **(A) `-march=native` 在 Apple clang 的 arm64 上 _不_ 启用 crypto 特性宏。** 最隐蔽的坑：`__ARM_FEATURE_SHA2` / `__ARM_NEON` 在裸 clang 默认 target 下有，加了 `-march=native` 反而消失，ARM SHA 路径永远走不到。必须用 **`-mcpu=native`**（保留全部 native 调优 + 启用 crypto）。Makefile 已按 `uname -m == arm64` 切换。验证：`echo | clang++ -mcpu=native -dM -E -x c++ - | grep ARM_FEATURE_SHA2`。
>
> **(B) ARM SHA-256 结构比 x86 SHA-NI 干净。** 无需 x86 那套 state shuffle（abef/cdgh 重排）。直接 `vsha256hq_u32` + `vsha256h2q_u32` 配对，`vsha256su0/su1` 做消息扩展，load/store 用 `vrev32q_u8` 做大端字节序转换。
>
> **(C) NEON 只有 128-bit → 4 路（x86 AVX2 是 256-bit → 8 路）。** RIPEMD 变量循环移位用 `vshlq_u32(x, vdupq_n_s32(n))` + `vshlq_u32(x, vdupq_n_s32(n-32))` 组合（NEON 负移位量 = 右移）。对称群加每组产点，攒 4 个一组喂给 `pubkey_to_hash160_4way`。

代码：`kernels/hash.h`（`sha256_33bytes_arm` / `ripemd160_4way`，gated on `__ARM_FEATURE_SHA2` / `__ARM_NEON`）。验证：ARM SHA 0/100k mismatch、NEON RMD 0/50k mismatch。

## 2.2 x86 CPU — 已落地 4.5x

设备 AMD EPYC Zen2。单线程 1.11 → 4.96 MK/s，8 核 8.8 → 39 MK/s。

| 优化 | 类型 | 单线程效果 |
|------|------|-----------|
| 对称群加 + 批量求逆（=§1.1） | 纯算法 | +118% |
| SHA-256 → SHA-NI 硬件指令 | x86 专属 | +34% |
| RIPEMD-160 → AVX2 8 路 SIMD | x86 专属 | +43% |
| mod_mul → MULX/ADCX/ADOX 双进位链 | x86 专属 | +7% |

代码：`kernels/hash.h`（gated on `__SHA__` / `__AVX2__`）、`kernels/secp256k1.h::mod_mul`（gated on `__BMI2__ && __ADX__`）。

## 2.3 负载均衡：动态自调度（平台无关，全 CPU 默认）

> 虽然在 ARM 工作中落地，但**与指令集无关，是所有 CPU 的正确默认**，x86/ARM 共用同一份纯 `std::thread`+`std::atomic` 代码。

**问题**：原来把 chunk 平均切 N 份（静态 push），总耗时被最慢的核绑架。

**方法**：范围切成大量 GROUP_SIZE 对齐的 tile，用一个原子计数器 `next_tile.fetch_add(1)` 分发，**谁先算完谁拉下一块**（pull 模型）。对所有"让核实际速度不一致"的因素自动适应：
- Apple P/E 核、**Intel 12 代起的 P/E 大小核**（E-core ≈ P-core 40-60%）
- Turbo 睿频、热降频、超线程（SMT）
- **某核被别的进程临时抢占**——它自然少拉 tile，其余核分摊，总耗时几乎不受影响

因为"按运行时实际产能拉取"而非"预先按核数推送"，对这类可独立切分的负载几乎严格优于静态均分，代价近乎零。

**tile 粒度量化（M3 Pro 实测）**：每 tile 固定开销 ≈ scalar_mul_g 重算 1015ns + 原子 30ns；稳态 ≈ 163 ns/key。两个相反损失（2³⁰ chunk、12 线程最坏估计）：

| tile | #tiles | 固定开销% | 长尾最坏% | 总% |
|------|--------|----------|----------|-----|
| GROUP×16 | 131k | 0.078 | 0.009 | 0.087 |
| **GROUP×32** | 65k | 0.039 | 0.018 | **0.057** ← 选用 |
| GROUP×64 | 33k | 0.020 | 0.037 | 0.056 |
| GROUP×256 | 8k | 0.005 | 0.147 | 0.152 |

U 型谷底在 GROUP×32~64。开销大头是**每 tile 重算一次中心点（比原子贵 33×）**,不是原子争用——所以 tile 不能太小;太大则长尾上升。选 GROUP×32（谷底 + 对小 `size` 更稳，每核保证 ≥8 块）。`size` 很小时需缩小 tile，代码里有该分支。

代码：`backend_cpu.h::search`。升级方向：当前单一共享计数器（tile 大、争用可忽略）；若未来 tile 需极小，可换 per-thread 本地队列 + 偷尾（rayon/TBB 式）。

## 2.4 Metal GPU — 已落地 12.4x

设备 M3 Pro。14.5 → 180 MK/s。完整路径：

| 阶段 | MK/s | 加速 | commit |
|------|------|------|--------|
| baseline | 14.5 | 1.0x | 6bedb7a |
| 增量点加 + 批量求逆 (S=8) | 37 | 2.5x | 429e93c |
| 调 KEYS_PER_THREAD → 128 | 83 | 5.7x | 22ea537 |
| 对称群加 C±i·G（§1.1） | 158 | 10.9x | symgroup |
| 调 GROUP_H → 256（§1.2） | **180** | **12.4x** | symgroup |

主要靠 §1 的算法优化。Metal 专属的微调（fast-math、threadgroup 宽度）收益不明显——见 §2.5 负优化记录。代码：`kernels/puzzle.metal`、`metal_solver.mm`。

## 2.5 已验证为负优化 / 无效（不要再试）

- **dedicated mod_sqr**（对称项算一次加两次）：Metal 83 → **61 MK/s，负优化**。原因：进位传播需分支/可变循环，伤 GPU 直线流水；通用 `mod_mul(a,a)` 的无分支固定结构反而更快。**CPU/Metal 都直接用 `mod_mul(a,a)`，不单独实现 mod_sqr。** 注意 `opt1-modsqr` 分支上还有一份**有 bug** 的手写 mod_sqr（129-bit doubling 移位错误，verify 失败），更不要用。
- **Metal threadgroup 宽度** 32/64/128/256：全部 ~81-83 MK/s，无差异。GPU 受 register/occupancy 限制，不是 threadgroup 调度限制。

---

# §3 可能的优化（未实现 / 进行中）

## 3.1 CUDA — 移植对称群加 ★最高优先级

**当前 CUDA kernel 落后整整一代算法。** `kernels/cuda/puzzle_kernel.cu` 注释写着 "extreme optimization for Hopper"，但那只是 **PTX 微优化**（carry-chain 内联汇编、mul.lo/hi、launch_bounds），**算法仍是最原始的 1 线程 = 1 key**（每 key 完整 scalar_mul + mod_inv）。

> **CUDA 最大的机会不是 PTX 微调，而是移植 §1.1 对称群加。** 预期同数量级提升（5-10x+）。PTX 那部分已够好，微调收益有限。

**移植要点**（详见 §1）：
- 蓝本 `backend_cpu.h::search_incremental`，几乎可一对一翻译
- **务必每线程一组、组内串行、不跨线程协作**（§1.3，这是被回退过的坑）
- H 从 16/32/64 往上 sweep（CUDA 寄存器比 Metal 更紧）

**架构 / 编译（P800 / A800 / H20 通用）**：
- 算法和基础 PTX（`mul.lo/hi.u64`、`add.cc`/`addc`）跨所有 NVIDIA 架构通用，没用 Hopper 专属 wgmma/TMA/cluster。**P800/A800 上调好的代码搬 H20 逻辑一字不改，只是 SM 更多跑更快。**
- 编译目标：`-DCMAKE_CUDA_ARCHITECTURES=`，Hopper(H20/H100/H800)=`90`，Ampere(A100/A800)=`80`，P800 确认架构后填对应 sm。
- `__launch_bounds__(256, 4)` 是给 sm_90 调的，换卡后重调。
- 正确姿势不跨线程，**不依赖大 shared memory**，P800 一样能拿到算法收益。

**落地前必须先建安全网（当前缺失！）**：CUDA 目前**没有任何正确性测试**。改 kernel 前必须照 `test_metal_correctness.cpp` 写 `test_cuda_correctness`：已知 key 放 batch 内非零偏移（1/37/1000/65535/1M），要求精确私钥 + 无误报，尤其覆盖 puzzle1(k=1) 的 degenerate fallback。

**建议顺序**：① 写 test_cuda_correctness（当前 1线程1key 版作 golden）→ ② 移植对称群加，每步过测试 → ③ sweep H → ④ sweep block size / launch_bounds → ⑤ 最后才碰 PTX 微调（dedicated mod_sqr 别试，见 §2.5）。

## 3.2 co-Z 加法（Meloni）—— 中优先级，与 §1.1 二选一

连续点共享 Z，增量步 ~5乘+2平方（vs Jacobian mixed-add 的 11乘+3平方）。比对称群加简单但收益略低。当前对称群加已落地，此项暂不需要。

## 3.3 低风险小优化（收益小，几乎无风险）

- scalar_mul_g 内层循环展开 / 预取
- 减少 affine 转换的中间变量寄存器占用
- 确认 SHA/RIPEMD 常量表在 constant 地址空间

---

# 安全网与测试

任何 kernel 改动后必跑。干净 main 为 golden。

| 测试 | 覆盖 | 命令 |
|------|------|------|
| `test_metal_correctness` | Metal 区间搜索（key 在非零偏移）+ degenerate + 无误报，9/9 | `make test-metal` |
| `test_coverage` | CPU 群搜索逐偏移穷举，1872/0 | `make test-coverage` |
| `test_correct` | CPU hash160 vs 已知 puzzle 真值 | `make test-correct` |
| `test_cuda_correctness` | **待写**（CUDA，见 §3.1） | — |
| 全部 | | `make test` |

实测性能口径：`./puzzle_worker --url http://<coordinator>:8080 --backend <name>`，看 per-task `MK/s`，**不看启动时的 init `Speed:`**（不准）。
