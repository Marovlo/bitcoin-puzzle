# Puzzle Worker

GPU/CPU 暴力搜索 worker。注册到协调者、批量拉取任务、计算 hash160、提交结果。

默认协调者：`http://81.70.166.231:8080`（公共池，无需配置）。

## 一键运行

### macOS / Linux

```bash
cd worker
./run.sh
```

`run.sh` 自动检测平台与 GPU，选择后端并构建，然后加入默认池：

| 环境 | 后端 | 构建方式 |
|------|------|---------|
| macOS | Metal | Makefile |
| Linux + `nvcc` | CUDA | CMake |
| Linux + `hipcc` | HIP (ROCm) | CMake |
| Linux（仅 CPU） | CPU | Makefile |
| Windows（AMD/NVIDIA/Intel） | OpenCL + CPU | `build_windows.ps1` |

常用参数（其余直接透传给 `puzzle_worker`）：

```bash
./run.sh --url http://其他协调者:8080   # 换协调者
./run.sh --backend cpu                   # 强制后端 auto|metal|cpu|cuda|hip
./run.sh --build-only                    # 只构建不运行
./run.sh --test                          # 构建并跑正确性测试
./run.sh --cpu-threads 16                # 透传给 worker
```

### Windows

```powershell
cd worker
.\build_windows.ps1              # 构建（WinLibs GCC + OpenCL）
.\build_windows.ps1 -Test        # 构建并跑正确性测试
.\build\puzzle_worker.exe --backend opencl --bench   # 测速，不连池
```

Windows 走 **OpenCL**（不需要任何 GPU SDK；kernel 由驱动运行时编译）。详见
[BUILD_WINDOWS.md](BUILD_WINDOWS.md)。

## 手动构建

```bash
make            # 构建 puzzle_worker（macOS=Metal, Linux=CPU）
make test       # 跑全部正确性测试
make clean      # 清理产物
./puzzle_worker --help
```

CUDA / HIP / OpenCL 需要 CMake（Makefile 只覆盖 Metal/CPU）：

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON   -DCMAKE_BUILD_TYPE=Release && make -j   # NVIDIA
cmake .. -DUSE_HIP=ON    -DCMAKE_BUILD_TYPE=Release && make -j   # AMD (ROCm, Linux)
cmake .. -DUSE_OPENCL=ON -DCMAKE_BUILD_TYPE=Release && make -j   # 任意 GPU（含 Windows/AMD）
```

详见 [BUILD_H20.md](BUILD_H20.md)（NVIDIA/CUDA）、[BUILD_ROCM.md](BUILD_ROCM.md)（AMD/ROCm，**Linux**）、
[BUILD_WINDOWS.md](BUILD_WINDOWS.md)（Windows，**OpenCL**）。

## 后端与性能

| 设备 | 后端 | 实测速率 |
|------|------|---------|
| Apple M3 Pro | Metal | ~180 MK/s |
| RTX 4090 | CUDA | ~2000-5000 MK/s |
| RTX 3070 Ti Laptop | CUDA | ~800-1500 MK/s |
| RX 6800 XT (Windows) | OpenCL | **~650 MK/s**（+CPU 合计 ~780） |
| RX 6800 XT (Windows，同卡) | CPU 16 线程 | ~63 MK/s |
| 多核 x86 | CPU | ~5 MK/s/核 |

> RX 6800 XT 在 Windows 上只能用 OpenCL：AMD 的 HIP SDK for Windows 官方只支持 RDNA3 及以上，
> gfx1030 被标为不支持。OpenCL 不需要任何 SDK，详见 [BUILD_WINDOWS.md](BUILD_WINDOWS.md)。

`--backend auto`（默认）会同时用 GPU + CPU。Metal 后端的优化历程见
[OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md)：增量点加 + Montgomery 批量求逆 +
对称群加 C±i·G，相对初版提升约 12.4x。

## 正确性测试

```bash
make test            # 全部
make test-correct    # CPU hash160 vs 已知 puzzle 真值
make test-coverage   # CPU 群搜索逐偏移穷举覆盖
make test-metal      # Metal 区间搜索（key 落在非零偏移处需精确命中）
```

Windows 上：

```powershell
.\build_windows.ps1 -Test    # test_correctness + test_opencl_correctness
```

`test_opencl_correctness` 覆盖：64 个偏移逐个精确命中（按 7 分块，跨批次边界）、
跨 2^64 的 128 位进位、2M key 的真实派发、以及两个反例（越界 key / 垃圾目标不得误报）。

## 文件结构

```
worker/
├── run.sh              # 一键构建+运行（macOS/Linux）
├── build_windows.ps1   # 一键构建+测试（Windows, MinGW GCC + OpenCL）
├── main.cpp            # pipeline 主程序（fetch | compute | submit 三线程）
├── worker.h            # ComputeBackend 接口
├── backend_multi.h     # 多后端并行（默认）
├── backend_metal.h     # Metal GPU
├── backend_cpu.h       # CPU 多线程（对称群搜索）
├── backend_cuda.h      # CUDA
├── backend_hip.h       # AMD HIP
├── backend_opencl.h    # OpenCL（Windows/AMD 主力）
├── http_client.h       # HTTP 通信
├── json_helpers.h      # JSON 解析
├── platform.h          # 平台兼容层（socket 类型 / poll / 超时）
├── Makefile            # Metal/CPU 构建 + 测试
├── CMakeLists.txt      # CUDA/HIP/OpenCL 构建
├── cmake/              # CMake 辅助脚本（内嵌 OpenCL kernel 源码）
├── third_party/OpenCL/ # Khronos 头文件（免装 SDK）
└── kernels/
    ├── secp256k1.h     # 有限域 + EC 运算（CPU）
    ├── hash.h          # SHA-256 + RIPEMD-160
    ├── puzzle.metal    # Metal GPU kernel
    ├── metal_solver.h/.mm
    ├── cpu_solver.h/.cpp
    ├── cuda/           # CUDA kernel
    ├── hip/            # HIP kernel
    └── opencl/         # OpenCL kernel + 宿主封装
```
