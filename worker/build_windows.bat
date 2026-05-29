@echo off
REM ============================================
REM Bitcoin Puzzle Worker - Windows One-Click Build
REM ============================================
REM Prerequisites:
REM   1. Visual Studio 2019/2022 with C++ Desktop workload
REM   2. CUDA Toolkit 12.x (https://developer.nvidia.com/cuda-downloads)
REM   3. CMake 3.18+ (https://cmake.org/download/ or via VS installer)
REM
REM Run this script from "x64 Native Tools Command Prompt for VS 2022"
REM or any terminal where cl.exe and nvcc are in PATH.
REM ============================================

echo [*] Checking prerequisites...

where cmake >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [!] CMake not found. Install from https://cmake.org/download/
    echo     Or enable it in Visual Studio Installer under "Desktop C++"
    exit /b 1
)

where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [!] nvcc not found. Install CUDA Toolkit from:
    echo     https://developer.nvidia.com/cuda-downloads
    echo.
    echo     After install, ensure these are in PATH:
    echo       C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
    echo.
    echo [*] Building CPU-only version instead...
    set USE_CUDA_FLAG=
    goto :build
)

echo [+] CUDA found: 
nvcc --version | findstr /i "release"
set USE_CUDA_FLAG=-DUSE_CUDA=ON

REM Auto-detect GPU architecture
echo [*] Detecting GPU architecture...
nvidia-smi --query-gpu=compute_cap --format=csv,noheader > %TEMP%\gpu_arch.txt 2>nul
if %ERRORLEVEL% equ 0 (
    set /p GPU_ARCH=<%TEMP%\gpu_arch.txt
    REM Convert "8.6" to "86"
    set GPU_ARCH=%GPU_ARCH:.=%
    echo [+] Detected GPU: sm_%GPU_ARCH%
    set ARCH_FLAG=-DCMAKE_CUDA_ARCHITECTURES="%GPU_ARCH%"
) else (
    echo [*] Cannot detect GPU, compiling for sm_86;89;90
    set ARCH_FLAG=-DCMAKE_CUDA_ARCHITECTURES="86;89;90"
)

:build
echo.
echo [*] Configuring with CMake...
if not exist build mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 %USE_CUDA_FLAG% %ARCH_FLAG%
if %ERRORLEVEL% neq 0 (
    echo [!] CMake configure failed.
    echo     Make sure you're running from VS Developer Command Prompt.
    exit /b 1
)

echo.
echo [*] Building Release...
cmake --build . --config Release --parallel
if %ERRORLEVEL% neq 0 (
    echo [!] Build failed.
    exit /b 1
)

echo.
echo ============================================
echo [+] BUILD SUCCESSFUL!
echo.
echo     Binary: build\Release\puzzle_worker.exe
echo.
echo     Quick test:
echo       build\Release\puzzle_worker.exe --test -u http://81.70.166.231:8080
echo.
echo     Run:
echo       build\Release\puzzle_worker.exe -u http://81.70.166.231:8080
echo ============================================
