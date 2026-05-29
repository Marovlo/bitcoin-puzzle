# Windows 构建指南（CUDA + CPU）

## 前置条件

1. **Visual Studio 2019/2022**（需要 C++ 桌面开发工作负载）
2. **CUDA Toolkit 12.x**：https://developer.nvidia.com/cuda-downloads
3. **CMake 3.18+**：https://cmake.org/download/

## 构建步骤

```powershell
cd bitcoin-puzzle\worker

# 创建构建目录
mkdir build
cd build

# 配置（启用 CUDA）
cmake .. -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON

# 编译 Release 版本
cmake --build . --config Release

# 可执行文件在
# build\Release\puzzle_worker.exe
```

### 指定 GPU 架构（可选，加速编译）

```powershell
# RTX 3070 Ti = SM 86 (Ampere)
cmake .. -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86"

# RTX 4090 = SM 89 (Ada Lovelace)
cmake .. -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89"
```

## 运行

```powershell
# 自检
.\Release\puzzle_worker.exe --test -u http://81.70.166.231:8080

# 正式运行
.\Release\puzzle_worker.exe -u http://81.70.166.231:8080

# 只用 CUDA GPU
.\Release\puzzle_worker.exe --backend cuda -u http://81.70.166.231:8080

# 只用 CPU（20 线程 for i7-12700H）
.\Release\puzzle_worker.exe --backend cpu --cpu-threads 20 -u http://81.70.166.231:8080
```

## 不装 CUDA（仅 CPU）

```powershell
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# 用法相同，只是没有 CUDA 后端
.\Release\puzzle_worker.exe --backend cpu --cpu-threads 20 -u http://server:8080
```

## 预期性能

| 设备 | 预期速率 |
|------|---------|
| RTX 3070 Ti Laptop | ~800-1500 MK/s |
| RTX 4090 | ~2000-5000 MK/s |
| i7-12700H (20 threads) | ~15-30 MK/s |

## 常见问题

### cl.exe 找不到
打开 "x64 Native Tools Command Prompt for VS 2022" 而不是普通 cmd。

### CUDA 编译很慢
添加 `-DCMAKE_CUDA_ARCHITECTURES="86"` 只编译你 GPU 的架构。

### 防火墙
确保 Windows 防火墙允许 puzzle_worker.exe 的出站 TCP 连接。
