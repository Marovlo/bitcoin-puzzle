# Metal 后端性能优化记录

分支 `opt-metal-perf`。设备：Apple M3 Pro。性能均为**实测**（连真实协调者跑任务的 per-task MK/s），非 init benchmark。

## 已落地（已验证 9/9 + 已提交）

| 阶段 | 真实 MK/s | 加速 | commit |
|------|-----------|------|--------|
| baseline (clean main) | 14.5 | 1.0x | 6bedb7a |
| 增量点加 + Montgomery 批量求逆 (S=8) | 37 | 2.5x | 429e93c |
| 调 KEYS_PER_THREAD → 128 | **83** | **5.7x** | 22ea537 |

**核心改动**：每个 GPU 线程处理 `KEYS_PER_THREAD` 个连续 key：
- `P_0 = k*G` 只算一次（windowed scalar_mul_g）
- `P_{i+1} = P_i + G` 用 Jacobian mixed-add 递推
- 所有点的 Z 坐标共享**一次** `mod_inv`（Montgomery trick）

摊薄了两个最贵操作（scalar_mul_g ~350 乘、mod_inv ~270 乘）。只复用已验证的域算术原语，未碰任何手写算术。

`KEYS_PER_THREAD` 扫描结果（S vs MK/s）：8→37, 16→50, 32→67, 64→79, **128→83**, 256→83（平台期）。128 是收益/栈占用的最佳点。

## 已验证为负优化 / 无效（不要再试）

- **dedicated mod_sqr**（对称项算一次加两次）：83 → **61 MK/s，负优化**。原因：进位传播需要分支/可变循环，伤 GPU 直线流水。通用 `mod_mul(a,a)` 的无分支固定结构反而更快。注意：`opt1-modsqr` 分支上还有一份**有 bug** 的手写 mod_sqr（129-bit doubling 移位错误，verify 失败），更不要用。
- **threadgroup 宽度** 32/64/128/256：全部 ~81-83 MK/s，无差异。GPU 受 register/occupancy 限制，不是 threadgroup 调度限制。

## 待尝试（未实现，预期有效，风险高）

### 1. 仿射增量加 + 批量分母求逆（最高优先级，预期再 1.4-1.9x → 110-150 MK/s）
当前瓶颈是 127 次 Jacobian `jac_add_mixed`（每次 ~11乘+3平方 ≈ 1780 域运算/线程，占大头）。

思路：每线程算 `start + i·G`，i=0..S-1。其中：
- `start = k_base·G`，用现有 scalar_mul_g 算一次并仿射化
- `i·G` 是 **host 端预计算的仿射常量表**（i=0..S-1），新开一个 MTLBuffer 上传
- 所有分母 `d_i = x(iG) − x(start)` 可**独立预算**（无串行依赖）→ 一次批量求逆得全部 1/d_i
- 每点加法只需 λ=(y(iG)−y(start))·(1/d_i)，x=λ²−x(start)−x(iG)，y=λ(x(start)−x)−y(start) ≈ 2乘+1平方

**正确性陷阱（必须处理）**：当 k_base = ±i 时 d_i=0（doubling/无穷远情形）。小 key 测试（puzzle1 k=1）必然触发。这些点必须走 Jacobian fallback，否则漏目标 key。9/9 区间测试能抓到这个 bug。

实现：host 端加 `build_step_table(S)` 生成 i·G 仿射表 + 新 buffer；kernel 重写主循环。

### 2. co-Z 加法（Meloni）替换 Jacobian mixed-add（中优先级）
连续点共享 Z，增量步 ~5乘+2平方（vs 11乘+3平方）。比方案 1 简单但收益略低。与方案 1 二选一。

### 3. 低风险小优化（收益小但几乎无风险）
- scalar_mul_g 内层循环展开 / 预取
- SHA256 / RIPEMD160 常量表用 constant 地址空间确认已最优
- 减少 affine 转换中间变量的寄存器占用

## 安全网
`worker/test_correctness.cpp`：区间搜索测试，把已知 key 放在 batch 内非零偏移（1, 37, 1000, 65535, 1M）处，要求返回精确私钥 + 无误报。任何 kernel 改动后必跑，干净 main 为 golden 9/9。

构建测试：
```
cd worker && make
clang++ -std=c++17 -O3 -march=native -I. -Ikernels -fobjc-arc \
  test_correctness.cpp metal_solver.o \
  -framework Metal -framework Foundation -framework CoreGraphics -lpthread \
  -o test_correctness && ./test_correctness
```

实测性能：`./puzzle_worker --url http://<coordinator>:8080 --backend metal`，看 per-task `MK/s`（不是开头的 init `Speed:`，那个不准）。
