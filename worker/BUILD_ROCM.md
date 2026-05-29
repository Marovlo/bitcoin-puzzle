# AMD ROCm (HIP) 构建指南

## 支持的 GPU

| GPU | 架构 | 代号 | 预期性能 |
|-----|------|------|---------|
| RX 6800 XT | RDNA2 | gfx1030 | ~500-1000 MK/s |
| RX 7900 XTX | RDNA3 | gfx1100 | ~800-1500 MK/s |
| MI250X | CDNA2 | gfx90a | ~2-4 GK/s |

## 前置条件

### Linux (推荐)
```bash
# 安装 ROCm (Ubuntu 22.04)
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm,hip

# 验证
rocminfo | grep "Name"
hipcc --version

# 确认用户在 video 和 render 组
sudo usermod -aG video,render $USER
```

### Windows (实验性)
```powershell
# 安装 HIP SDK: https://www.amd.com/en/developer/resources/rocm-hub.html
# RX 6800 XT 需要 HIP SDK 5.7+
# 确认 hipcc.exe 在 PATH 中
```

## 构建

```bash
cd bitcoin-puzzle/worker
mkdir build && cd build

# CMake 构建
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1030" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 或直接 hipcc 编译（简单场景）
hipcc -O3 -std=c++17 --offload-arch=gfx1030 \
  -DUSE_HIP -I.. -Ikernels \
  main.cpp kernels/hip/puzzle_kernel.hip kernels/hip/hip_solver.hip \
  -o puzzle_worker -lpthread
```

### 架构参数对照

| GPU | --offload-arch |
|-----|---------------|
| RX 6800 XT | gfx1030 |
| RX 6900 XT | gfx1030 |
| RX 7900 XTX | gfx1100 |
| RX 7600 | gfx1102 |
| MI250X | gfx90a |

## 运行

```bash
# 自检
./puzzle_worker --test -u http://81.70.166.231:8080

# 正式运行
./puzzle_worker -u http://81.70.166.231:8080

# 只用 HIP GPU
./puzzle_worker --backend hip -u http://81.70.166.231:8080
```

## CMakeLists.txt 变更

在 CMakeLists.txt 中添加 HIP 支持：

```cmake
option(USE_HIP "Enable AMD ROCm/HIP support" OFF)
if(USE_HIP)
    enable_language(HIP)
    add_definitions(-DUSE_HIP)
    list(APPEND WORKER_SOURCES
        kernels/hip/puzzle_kernel.hip
        kernels/hip/hip_solver.hip
    )
endif()
```

## 常见问题

### "hipErrorNoBinaryForGpu"
编译时的 `--offload-arch` 和你实际 GPU 不匹配。运行 `rocminfo | grep gfx` 查看。

### 性能低于预期
```bash
# 检查 GPU 状态
rocm-smi

# 检查是否被功耗限制
rocm-smi --showpower

# 强制最大性能模式
rocm-smi --setperflevel high
```

### Windows 上 gfx1036 误识别
已知 bug：ROCm 6.1/6.2 可能把 RX 6800 XT 误识别为 gfx1036。
解决：设置环境变量 `HSA_OVERRIDE_GFX_VERSION=10.3.0`
