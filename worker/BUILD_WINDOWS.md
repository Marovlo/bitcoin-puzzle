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
| `--backend opencl` | **~680 MK/s** | 群搜索 kernel，整机 CPU 占用约 2% |
| `--backend cpu` | ~63 MK/s | 16 线程，整机 CPU 满载 |
| `--backend auto` | ~590–716 MK/s | GPU + CPU 并行，长跑抖动较大 |

作为对比，同一张卡上朴素实现（每 key 一次标量乘 + 一次求逆）只有 ~43 MK/s。

> 长跑建议只用 GPU。`auto` 会额外占满 16 个 CPU 线程（约 110 W），但实测**长跑吞吐并不稳定优于 GPU-only**，
> 所以在这台机器上 `--backend opencl` 是更合适的默认值。

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

## 功耗控制（AMD / OverDrive8）

Windows 没有 GPU 功耗计数器（`Power Meter` 是电池的），而 Adrenalin 的功耗滑杆在不少 RX 6000
AIB 卡上**范围只有 -6%..+15%**，最低也就到 ~235 W，达不到真正的低功耗目标。
真正有效的杠杆是**限制 GFX 最高频率**——降频时驱动会把电压一起压到 881 mV，省电主要来自这里。

`tools/amd_power.exe` 通过驱动的 ADL OverDrive8 接口读写这些项（与 Adrenalin 调校页滑杆同一条路），
并用 PMLog 读出真实瓦数 / 温度 / 转速。

```powershell
.\tools\amd_power.exe info              # 能力位、各项可调范围、实时传感器
.\tools\amd_power.exe watch 10          # 每秒采一次功耗/温度/转速
.\tools\amd_power.exe set-clkmax 1100   # 限制最高频率 (MHz)
.\tools\amd_power.exe set-power -6      # 功耗滑杆 (%)
.\tools\amd_power.exe reset             # 恢复默认
```

RX 6800 XT 实测（worker 满载，`--backend opencl`，`ASIC_POWER` 单位为 W）：

| GFX 频率上限 | 实际频率 | 功耗 | 算力 | 每瓦算力 |
|---|---|---|---|---|
| 默认 2519 | 2360 MHz | 244 W | 682 MK/s | 2.79 |
| 2200 | 2165 MHz | 164 W | 595 MK/s | 3.63 |
| **1900** | **1874 MHz** | **131 W** | **516 MK/s** | **3.94 ← 效率最高** |
| 1600 | 1585 MHz | 120 W | 436 MK/s | 3.63 |
| 1300 | 1284 MHz | 109 W | 355 MK/s | 3.26 |
| **1100** | **1093 MHz** | **99 W** | **302 MK/s** | **3.05** |
| 900 | 899 MHz | 91 W | 248 MK/s | 2.73 |
| 700 | 699 MHz | 70 W | 193 MK/s | 2.75 |

关于这张表：

- **算力与实际频率严格线性**（约 0.276 MK/s 每 MHz），所有行都吻合，所以表里的算力不是估算值。
- **功耗曲线很不线性**：2470 → 2170 MHz 一段掉了 85 W，因为电压从 1128 mV 掉到 950 mV；
  到 1900 MHz 以下电压触到 881 mV 地板，之后功耗只随频率缓降。
- **每瓦算力在 ~1900 MHz 最优（3.94）**。因为电压有地板，频率太低反而摊不平固定的电压成本。
  所以「要省电」和「要效率」是两个不同的点：
  - 只要功耗 ≤100 W → `set-clkmax 1100`（99 W，热点 62 °C，风扇约 975 rpm，有时直接停转）；
  - 想要**最高每瓦算力** → `set-clkmax 1900`（131 W，516 MK/s，比默认省电 46% 而算力只掉 24%）。
- 满载功耗地板约 70 W（700 MHz），再往下压收益很小。

相比默认 2519 MHz：1100 MHz 下 **功耗 -59%、热点 -22 °C、风扇转速 -44%，算力 -56%**；40 秒采样内
功耗稳定在 98–101 W，无漂移。

### 让限制在重启后保持

OverDrive8 设置属于运行时状态，**驱动重启后会回到默认**。开机自动应用（无需管理员权限）：
把下面内容存为
`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\amd-gpu-100w.vbs`：

```vbs
Set sh = CreateObject("WScript.Shell")
sh.Run """<仓库路径>\worker\tools\amd_power.exe"" set-clkmax 1100", 0, False
```

第二个参数 `0` 表示隐藏窗口。删掉该文件即可取消。

### 这张卡上其它可用的旋钮

`amd_power.exe info` 会列出全部可调项及范围。除频率外还可用：

- `OD_VOLTAGE`（881–1150 mV，电压上限）——在降频之外再压一档电压；
- `FAN_CURVE_*`（风扇曲线）——用 `set k=v` 自定义更安静的曲线。

能力位未置位、即**不可用**的有：`FAN_ACOUSTIC_LIMIT`（封顶转速）、`TEMPERATURE_FAN`（目标温度）、
`GFX_VOLTAGE_LIMIT`、`TDC_LIMIT`。

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
- 若设置了 GFX 频率上限（见「功耗控制」），算力会按频率成比例下降，这是预期行为。
- 其它 3D 应用（游戏、浏览器硬件加速、壁纸引擎）会抢占 GPU，关掉再测。
