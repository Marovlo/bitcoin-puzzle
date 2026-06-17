# Bitcoin Puzzle Pool

分布式比特币 puzzle 暴力搜索矿池。学习 secp256k1、Metal GPU、分布式系统的教学项目。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                Coordinator (Go + SQLite)                     │
│                                                             │
│  Random allocation:                                         │
│    - crypto/rand 生成随机 chunk index                        │
│    - DB UNIQUE 约束保证无重复                                 │
│    - 只存 issued 集合（< 0.001% of total），不会爆             │
│    - 10min 超时自动回收 stale tasks                          │
│                                                             │
│  Storage: O(completed_tasks)，不是 O(total_chunks)           │
│    - Puzzle #71: 跑 1 年 ~3M 条 = ~24MB                     │
│    - Puzzle #71: 全空间 2^40 条 若全存 = 8TB ← 我们不存这个     │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP JSON (batch)
     ┌─────────────────┼─────────────────┐
     │                 │                 │
┌────▼───┐  ┌─────────▼──┐  ┌──────────▼──┐
│Fetch   │  │  Compute   │  │  Submit     │
│Thread  │→ │  Thread    │→ │  Thread     │  ← Worker (pipeline)
│        │  │(Metal+CPU) │  │(batch POST) │
└────────┘  └────────────┘  └─────────────┘
  pre-fetch    never waits    async flush
  task_queue   on network     result_queue
```

## 核心设计解答

### Q: task 太多会不会 SQLite 存储爆炸？
**不会。** 我们只存 "已发出" 的 task（active + done），不存"所有可能的"。  
- Puzzle #71 总 chunk 数 = 2^40 ≈ 1万亿
- 但实际完成的：1年 × 100 workers × 10MK/s ÷ 2^30 keys/chunk ≈ **3M 条 = 24MB**
- SQLite 轻松承载

### Q: 随机选择会不会因碰撞成为瓶颈？
**不会。** 碰撞概率 = completed / total_chunks。  
- 完成 3M / 2^40 ≈ 0.0003%，期望 1 次随机就命中空闲 chunk
- 即使完成 50%，平均 2 次随机即可（远比 DB scan 快）
- 极端情况（>99% 完成）：回退到 reclaim stale 或顺序扫描

### Q: 高并发 + 数据一致性？
- **Mutex** 保护整个分配路径（`allocateChunks`）
- SQLite `UNIQUE INDEX` 做最终兜底：即使两个 worker 同时拿到相同 index，INSERT OR IGNORE 只有一个成功
- 单 writer（SQLite 特性），不存在写写冲突

### Q: 快设备通信瓶颈？
- **Batch fetch**: 一次拉取 N 个 task，Worker 自动校准 N
- **Pipeline**: fetch/compute/submit 三线程，compute 永远不等网络
- **Auto-calibrate**: 首个 task 完成后计算理想 batch size（目标 ~5 min/batch）

## 快速开始

```bash
# 1. Coordinator
cd coordinator && go build && PUZZLE_NUM=20 ./puzzle_coordinator

# 2. Worker (auto-detects Metal + CPU)
cd worker && make && ./puzzle_worker --url http://localhost:8080

# 3. Stats
curl http://localhost:8080/api/stats | jq
```

## 项目结构

```
bitcoin-puzzle/
├── coordinator/
│   ├── main.go              # 协调者（任务分配 + 持久化）
│   └── go.mod
├── worker/
│   ├── main.cpp             # Worker pipeline 主程序
│   ├── worker.h             # ComputeBackend 接口
│   ├── backend_multi.h      # 多设备并行（默认）
│   ├── backend_cpu.h        # CPU 多线程后端
│   ├── backend_metal.h      # Metal GPU 后端
│   ├── backend_cuda.h       # CUDA 存根
│   ├── http_client.h        # HTTP 通信
│   ├── json_helpers.h       # JSON 解析
│   ├── Makefile
│   └── kernels/
│       ├── secp256k1.h      # EC 有限域运算
│       ├── hash.h           # SHA-256 + RIPEMD-160
│       ├── puzzle.metal     # Metal GPU kernel
│       ├── metal_solver.h/mm # Metal 调度器
│       ├── cpu_solver.h/cpp  # 单线程 CPU
│       └── types.h
├── puzzles.json             # 160 个 puzzle 数据
└── README.md
```

## API

| 端点 | 说明 |
|------|------|
| `POST /api/register` | Worker 注册 |
| `GET /api/tasks?worker_id=x&count=N` | 批量获取 N 个 task |
| `POST /api/submit` | 批量提交结果 `{results: [...]}` |
| `GET /api/stats` | 矿池状态 |

## 性能 (Apple M3 Pro)

| 后端 | 速率 | 角色 |
|------|------|------|
| Metal GPU | ~83 MK/s | 主力 |
| CPU 多线程 | ~5 MK/s | 辅助 |
| multi[metal+cpu] | ~88 MK/s | 默认 |

> Metal 实测（连协调者跑真实任务，非 init benchmark）。优化历程见
> `worker/OPTIMIZATION_NOTES.md`：增量点加 + Montgomery 批量求逆 +
> KEYS_PER_THREAD 调优，相对初版 14.5 MK/s 提升约 5.7x。

## 免责声明

教学项目。实际解 puzzle #71+ 需要数百万 GPU·年。
