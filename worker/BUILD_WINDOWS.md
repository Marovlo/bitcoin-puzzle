# Windows 构建指南（OpenCL GPU + CPU）

Windows 上走 **OpenCL**，不是 CUDA/HIP。原因：

- AMD 的 **HIP SDK for Windows 官方只支持 RDNA3 及以上**，RX 6800 XT（gfx1030，RDNA2）被明确标注为 ❌ 不支持，装 SDK 也跑不起来；
- OpenCL 随**显卡驱动**一起安装，**不需要任何 SDK**，而且 kernel 由驱动在运行时编译，所以构建时只需要一个宿主编译器；
- 同一套代码在 NVIDIA / Intel 显卡上也能用（驱动提供的 OpenCL 平台不同而已）。

## 前置条件

只需要一个 C++ 编译器。推荐 **WinLibs GCC**（免安装器，自带 cmake/ninja/gdb）：

```powershell
winget install BrechtSanders.WinLibs.POSIX.UCRT
```

> **为什么不用 MSVC？** `kernels/secp256k1.h` 用了 `unsigned __int128`，而 MSVC 至今不支持该类型，CPU 后端用 `cl.exe` 编译不过。

无需手动安装 OpenCL 头文件——仓库已内置 Khronos 官方头文件于 `third_party/OpenCL/`；
链接目标 `OpenCL.dll`（ICD loader）由显卡驱动安装到 `C:\Windows\System32`。

## 构建

```powershell
cd worker
.\build_windows.ps1            # 构建 puzzle_worker.exe + 测试
.\build_windows.ps1 -Test      # 构建并跑全部正确性测试
.\build_windows.ps1 -Clean     # 清理产物
```

产物在 `worker\build\`。

## 运行

先确认驱动被识别：

```powershell
.\build\puzzle_worker.exe --list-devices
```
```
=== Available GPU devices ===
OpenCL:
  Platform: AMD Accelerated Parallel Processing
  [0] gfx1030 (36 CUs, 16368 MB) OpenCL 2.0 AMD-APP (3652.0)
```

测速（**不会连接任何协调者**）：

```powershell
.\build\puzzle_worker.exe --backend opencl --bench   # 只 GPU
.\build\puzzle_worker.exe --backend cpu    --bench   # 只 CPU
.\build\puzzle_worker.exe --backend auto   --bench   # GPU + CPU
```

自检并加入公共池：

```powershell
# 自检（跑正确性测试，不加入池）
.\build\puzzle_worker.exe --test -u http://81.70.166.231:8080

# 正式运行
.\build\puzzle_worker.exe -u http://81.70.166.231:8080
```

## 实测性能（RX 6800 XT + Ryzen 7 5700X，50M key 批量）

| 后端 | 速率 | 说明 |
|------|------|------|
| `--backend opencl` | **~645 MK/s** | 群搜索 kernel |
| `--backend cpu` | ~63 MK/s | 16 线程 |
| `--backend auto` | **~780 MK/s** | GPU + CPU 并行 |

作为对比，同一张卡上朴素实现（每 key 一次标量乘 + 一次求逆）只有 ~43 MK/s。

## 用 CMake 构建（可选）

```powershell
cd worker
cmake -S . -B build-cmake -G "MinGW Makefiles" -DUSE_OPENCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake -j
```

## 调参

| 环境变量 | 默认 | 作用 |
|---------|------|------|
| `PUZZLE_OCL_BATCH` | 16000000 | 每次派发的 key 数。派发开销按次数计，批量太小会被驱动调用淹没 |
| `PUZZLE_OCL_NAIVE` | 未设置 | 设为任意值则强制使用朴素 kernel，仅用于 A/B 对照 |

## 常见问题

### 找不到 GPU / `No GPU OpenCL platform found`

驱动没装好或版本过旧。用 `clinfo.exe` 验证（AMD 驱动自带，位于
`C:\Windows\System32\DriverStore\FileRepository\amdocl.inf_amd64_*\`）：

```powershell
& "C:\Windows\System32\DriverStore\FileRepository\amdocl.inf_amd64_564a246131a7b2c6\clinfo.exe"
```

应能看到 `AMD Accelerated Parallel Processing` 平台与你的显卡。

### 编译报 `unsigned __int128` 相关错误

用到了 MSVC。请改用本页的 WinLibs GCC 流程。

### Clash / 代理

worker 连协调者是普通 HTTP。若开了系统代理，注意 `--url` 指向的主机可能需要绕过代理。

### 防火墙

确保 Windows 防火墙允许 `puzzle_worker.exe` 的出站 TCP 连接。

### 性能低于预期

- 确认用的是群搜索 kernel：启动日志里 `kernel group (group 32+1 keys, 8 groups/item)`；
  显示 `kernel naive` 说明设置了 `PUZZLE_OCL_NAIVE`。
- `--backend auto` 会把 GPU 和 CPU 一起用上，比单用 GPU 更快。
- 其它 3D 应用（游戏、浏览器硬件加速、壁纸引擎）会抢占 GPU，关掉再测。
