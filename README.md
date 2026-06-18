# Bitcoin Puzzle Pool

分布式比特币 puzzle 暴力搜索矿池。学习 secp256k1、GPU 计算、分布式系统的教学项目。

一个 **coordinator**（Go）分发搜索区间，任意数量的 **worker**（C++，支持 Metal / CUDA / HIP / CPU）领取区间、计算、提交结果。下面教你在自己的设备上把 worker 跑起来，加入公共池。

---

## 在你的设备上跑起来

worker 默认连接公共协调者 `http://81.70.166.231:8080`，**无需任何配置**。先按你的平台装好依赖，然后一行命令构建并加入。

### macOS（Apple Silicon / Intel）

```bash
# 依赖：Xcode 命令行工具（含 clang、Metal）
xcode-select --install        # 如未安装

git clone https://github.com/Marovlo/bitcoin-puzzle.git
cd bitcoin-puzzle/worker
./run.sh                      # 自动检测 Metal GPU + CPU，构建并加入
```

### Linux — NVIDIA GPU（CUDA）

```bash
# 依赖：CUDA Toolkit（提供 nvcc）、cmake、g++
#   Ubuntu: sudo apt install nvidia-cuda-toolkit cmake build-essential
git clone https://github.com/Marovlo/bitcoin-puzzle.git
cd bitcoin-puzzle/worker
./run.sh                      # 检测到 nvcc -> 自动用 CUDA 构建
```

### Linux — AMD GPU（ROCm/HIP）

```bash
# 依赖：ROCm（提供 hipcc）、cmake
git clone https://github.com/Marovlo/bitcoin-puzzle.git
cd bitcoin-puzzle/worker
./run.sh                      # 检测到 hipcc -> 自动用 HIP 构建
```

### Linux — 仅 CPU（无 GPU）

```bash
# 依赖：g++/clang++、make
git clone https://github.com/Marovlo/bitcoin-puzzle.git
cd bitcoin-puzzle/worker
./run.sh                      # 无 GPU 工具链 -> CPU 多线程构建
```

### Windows（NVIDIA / CPU）

走 CMake + Visual Studio，步骤见 [worker/BUILD_WINDOWS.md](worker/BUILD_WINDOWS.md)。

> `run.sh` 会自动检测平台与 GPU、选对构建方式（Metal/CPU 用 Makefile，CUDA/HIP 用 CMake），编译后直接加入公共池。Ctrl+C 停止。

---

## 验证与监控

```bash
# 先自检（构建 + 跑正确性测试，不加入池）
./run.sh --test

# 只构建不运行
./run.sh --build-only

# 看公共池状态（已搜索 key 数、完成块数、在线 worker 数）
curl http://81.70.166.231:8080/api/stats
```

启动后每完成一个任务会打印实测速率，例如：

```
  Backend:   multi[metal+cpu]
  [1] chunk=316727949467 5.4s 197.8 MK/s q=24
```

---

## 常用选项

`run.sh` 把额外参数透传给 `puzzle_worker`，常用的：

```bash
./run.sh --backend metal        # 强制后端：auto(默认) | metal | cpu | cuda | hip
./run.sh --backend cpu --cpu-threads 16   # 限制 CPU 线程数
./run.sh --url http://其他主机:8080        # 连别的协调者
./puzzle_worker --help          # 查看全部参数
```

默认 `--backend auto`（=`multi`）同时使用 GPU + CPU。更多用法、各 GPU 构建细节、调参见 [worker/README.md](worker/README.md)。

---

## 性能 (Apple M3 Pro，实测)

| 后端 | 速率 | 角色 |
|------|------|------|
| Metal GPU | ~180 MK/s | 主力 |
| CPU 12 线程 | ~46 MK/s | 辅助 |
| multi[metal+cpu] | ~225 MK/s | 默认 |

| 设备 | 后端 | 预期速率 |
|------|------|---------|
| RTX 4090 | CUDA | ~2000-5000 MK/s |
| RTX 3070 Ti Laptop | CUDA | ~800-1500 MK/s |
| Apple M3 Pro | Metal+CPU | ~225 MK/s |

> 数值为实测（连协调者跑真实任务，非启动时的 init benchmark）。GPU/CPU 的优化历程与方法见 [worker/OPTIMIZATION_NOTES.md](worker/OPTIMIZATION_NOTES.md)：
> - **Metal**：增量点加 + Montgomery 批量求逆 + 对称群加 C±i·G，14.5 → 180 MK/s（12.4x）
> - **CPU**：动态自调度负载均衡（适配大小核/睿频/抢占）+ 平台专属哈希指令（x86 SHA-NI/AVX2、ARM SHA2/NEON）

---

## 自建协调者（可选）

只想加入公共池的话不需要这步。要自己搭一个：

```bash
# 1. 启动 coordinator（PUZZLE_NUM 选目标 puzzle 编号）
cd coordinator && go build && PUZZLE_NUM=20 ./puzzle_coordinator

# 2. worker 指向它
cd worker && ./run.sh --url http://localhost:8080

# 3. 看状态
curl http://localhost:8080/api/stats | jq
```

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                Coordinator (Go + SQLite)                     │
│  - crypto/rand 生成随机 chunk index，DB UNIQUE 约束保证无重复   │
│  - 只存 issued 集合（< 0.001% of total），10min 超时回收 stale  │
│  - 存储 O(completed_tasks) 而非 O(total_chunks)               │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP JSON (batch)
     ┌─────────────────┼─────────────────┐
┌────▼───┐  ┌─────────▼──┐  ┌──────────▼──┐
│Fetch   │  │  Compute   │  │  Submit     │  ← Worker (pipeline)
│Thread  │→ │  Thread    │→ │  Thread     │
│pre-fetch│  │(GPU+CPU)   │  │(batch POST) │
└────────┘  └────────────┘  └─────────────┘
  task_queue   永不等网络      result_queue
```

worker 是 fetch / compute / submit 三线程流水：预拉任务、计算永不等网络、结果异步批量提交。

### 设计要点

- **存储不爆**：只存"已发出"的 task（active + done）。Puzzle #71 跑一年约 3M 条 ≈ 24MB，SQLite 轻松承载；不存全空间的 2^40 条。
- **随机分配无瓶颈**：碰撞概率 = completed / total_chunks，完成 3M / 2^40 ≈ 0.0003%，期望 1 次随机即命中空闲 chunk。极端（>99% 完成）回退到 reclaim stale。
- **一致性**：分配路径 Mutex 保护 + SQLite `UNIQUE INDEX` 兜底（`INSERT OR IGNORE` 保证两个 worker 拿到同一 index 时只有一个成功）。
- **快设备不被网络拖慢**：batch fetch + 三线程 pipeline + 首任务后自动校准 batch size（目标 ~5 min/batch）。

## API

| 端点 | 说明 |
|------|------|
| `POST /api/register` | Worker 注册 |
| `GET /api/tasks?worker_id=x&count=N` | 批量获取 N 个 task |
| `POST /api/submit` | 批量提交结果 `{results: [...]}` |
| `GET /api/stats` | 矿池状态 |

## 项目结构

```
bitcoin-puzzle/
├── coordinator/         # 协调者（Go）：任务分配 + 持久化
│   └── main.go
├── worker/              # worker（C++）：多后端搜索
│   ├── run.sh           # 一键构建+运行（macOS/Linux）
│   ├── README.md        # worker 详细文档
│   ├── main.cpp         # pipeline 主程序
│   ├── backend_*.h      # multi / metal / cpu / cuda / hip 后端
│   ├── Makefile         # Metal/CPU 构建 + 测试
│   ├── CMakeLists.txt   # CUDA/HIP 构建
│   ├── OPTIMIZATION_NOTES.md
│   └── kernels/         # secp256k1 / hash / 各平台 GPU kernel
├── puzzles.json         # puzzle 数据
└── README.md
```

## 免责声明

教学项目。实际解 puzzle #71+ 需要数百万 GPU·年——本项目用于学习椭圆曲线密码、GPU/SIMD 优化与分布式系统，不期望真的解出。
