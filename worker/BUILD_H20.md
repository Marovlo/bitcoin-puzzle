# H20 GPU 部署指南

## 设备信息

- NVIDIA H20: Hopper 架构, sm_90, 60 SMs, 96GB HBM3
- 预期性能: **3-8 GKeys/s**（约 M3 Pro 的 300-700 倍）

## 前置条件

```bash
# 确认 GPU 可见
nvidia-smi

# 确认 CUDA Toolkit（需要 12.0+）
nvcc --version

# 如果没有 CUDA Toolkit:
# Ubuntu:
sudo apt install -y nvidia-cuda-toolkit

# 或从官网安装（推荐最新版）:
# https://developer.nvidia.com/cuda-downloads?target_os=Linux
```

## 构建

```bash
# 1. 拉取代码
git clone https://github.com/Marovlo/bitcoin-puzzle.git
cd bitcoin-puzzle/worker

# 2. 用 CMake 构建（指定 sm_90 for Hopper）
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 编译约 1-2 分钟（CUDA kernel 编译较慢）
```

## 运行

```bash
# 自检
./puzzle_worker --test -u http://81.70.166.231:8080

# 正式运行（H20 全速）
./puzzle_worker -u http://81.70.166.231:8080

# 后台运行
nohup ./puzzle_worker -u http://81.70.166.231:8080 > puzzle.log 2>&1 &
tail -f puzzle.log

# 如果机器有多个 GPU，跑多个 Worker 实例（每个绑定不同 GPU）
# 目前自动使用 device 0，多 GPU 支持待开发
```

## 性能调优

```bash
# 调大 batch（H20 可以轻松处理 32M-64M keys/dispatch）
./puzzle_worker --metal-batch 32000000 -u http://server:8080
# 注：--metal-batch 参数对 CUDA 也生效（统一的 batch 参数）

# 如果不想用 CPU（省电）
./puzzle_worker --backend cuda -u http://server:8080

# GPU + CPU 全部用上（默认行为）
./puzzle_worker -u http://server:8080
```

## 预期输出

```
  [+] CUDA GPU: NVIDIA H20 (SM 9.0, 60 SMs, 93804 MB)
  [+] CPU (96 threads): available
  [*] Benchmarking devices...
      cuda: 5000000000 Keys/s (5000.00 MK/s)
      cpu:  50000000 Keys/s (50.00 MK/s)
=== Bitcoin Puzzle Pool Worker (Pipeline) ===
  Speed:     5050.00 MKeys/s
  Registered OK

[*] Pipeline started: fetch | compute | submit
  [fetch] Target updated: puzzle #71 h160=f6f5431d...
  [1] chunk=xxx 0.2s 5120.0 MK/s q=50
  [auto] batch_count=50 (0.2s/task)
```

## 多 GPU 部署（如果有多卡）

```bash
# 每卡一个进程
CUDA_VISIBLE_DEVICES=0 ./puzzle_worker --id worker_gpu0 -u http://server:8080 &
CUDA_VISIBLE_DEVICES=1 ./puzzle_worker --id worker_gpu1 -u http://server:8080 &
CUDA_VISIBLE_DEVICES=2 ./puzzle_worker --id worker_gpu2 -u http://server:8080 &
# ...
```

## 常见问题

### "No CUDA device found"
```bash
nvidia-smi  # 确认驱动正常
ls /dev/nvidia*  # 确认设备文件存在
```

### 编译时 "unsupported gpu architecture sm_90"
CUDA Toolkit 版本太老。需要 CUDA 12.0+ 才支持 sm_90。

### 性能低于预期
```bash
# 检查 GPU 时钟频率
nvidia-smi -q -d CLOCK

# 检查是否功耗受限
nvidia-smi -q -d POWER

# 设置持久模式
sudo nvidia-smi -pm 1
```
