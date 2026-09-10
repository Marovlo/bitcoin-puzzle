# Windows build script for the puzzle worker (MinGW-w64 GCC + OpenCL).
#
# Why MinGW/GCC rather than MSVC:
#   secp256k1.h uses `unsigned __int128`, which MSVC does not implement, so the
#   CPU backend cannot be built with cl.exe as-is.
#
# Why OpenCL rather than CUDA/HIP:
#   AMD's HIP SDK for Windows supports RDNA3 and newer only; RX 6800 XT
#   (gfx1030, RDNA2) is marked unsupported. OpenCL ships with the normal
#   Adrenalin driver and compiles the kernel at runtime, so no extra SDK is
#   needed at all.
#
# Usage (from the worker/ directory):
#   .\build_windows.ps1              # build puzzle_worker.exe + tests
#   .\build_windows.ps1 -Test        # build then run the correctness tests
#   .\build_windows.ps1 -Clean       # remove build outputs

[CmdletBinding()]
param(
    [switch]$Test,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Push-Location $root
try {
    $outDir = Join-Path $root 'build'
    if ($Clean) {
        Remove-Item -Recurse -Force $outDir -ErrorAction SilentlyContinue
        Remove-Item -Force (Join-Path $root 'kernels\opencl\puzzle_opencl_source.inc') -ErrorAction SilentlyContinue
        Write-Host 'cleaned.' -ForegroundColor Green
        return
    }

    # ---- locate the toolchain -------------------------------------------------
    function Find-Gxx {
        $onPath = Get-Command 'g++.exe' -ErrorAction SilentlyContinue
        if ($onPath) { return $onPath.Source }

        # winget WinLibs layout
        $candidates = @(
            "$env:LOCALAPPDATA\Microsoft\WinGet\Packages",
            'C:\Program Files\WinGet\Packages',
            'C:\mingw64', 'D:\mingw64', 'C:\msys64\mingw64'
        )
        foreach ($base in $candidates) {
            if (-not (Test-Path $base)) { continue }
            $found = Get-ChildItem -Path $base -Recurse -Filter 'g++.exe' -ErrorAction SilentlyContinue |
                     Select-Object -First 1
            if ($found) { return $found.FullName }
        }
        throw 'g++ not found. Install it with: winget install BrechtSanders.WinLibs.POSIX.UCRT'
    }

    $gxx = Find-Gxx
    $gxxDir = Split-Path $gxx -Parent
    $env:PATH = "$gxxDir;$env:PATH"
    Write-Host "using $gxx" -ForegroundColor Cyan
    & $gxx --version | Select-Object -First 1

    # ---- locate the OpenCL ICD loader ----------------------------------------
    $openclDll = 'C:\Windows\System32\OpenCL.dll'
    if (-not (Test-Path $openclDll)) { throw "OpenCL.dll not found at $openclDll" }

    # ---- embed the kernel source ---------------------------------------------
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $clSrc = Join-Path $root 'kernels\opencl\puzzle_kernel.cl'
    $incDst = Join-Path $root 'kernels\opencl\puzzle_opencl_source.inc'
    $kernel = Get-Content $clSrc -Raw
    # Raw string literal: R"OPENCL( ... )OPENCL". The delimiter cannot occur in
    # the kernel source, so no escaping is required.
    # WriteAllText with an explicit BOM-less encoding: PowerShell's -Encoding UTF8
    # prepends a BOM, which would corrupt the literal.
    $literal = "R`"OPENCL(" + $kernel + ")OPENCL`""
    [System.IO.File]::WriteAllText($incDst, $literal, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "embedded kernel -> kernels/opencl/puzzle_opencl_source.inc" -ForegroundColor Cyan

    # ---- compile --------------------------------------------------------------
    # -march=native unlocks AVX2 / BMI2 / ADX / SHA-NI on Zen 3, which hash.h
    # and secp256k1.h detect via feature macros.
    $cxxflags = @(
        '-std=c++17', '-O3', '-DNDEBUG', '-march=native',
        '-DUSE_OPENCL',
        '-I.', '-Ikernels', '-Ithird_party/OpenCL',
        '-Wall', '-Wextra', '-Wno-unused-parameter'
    )
    # -static pulls in libstdc++/libgcc/libwinpthread so the .exe runs anywhere
    # without shipping MinGW DLLs alongside it.
    $ldflags = @('-static', $openclDll, '-lws2_32')

    function Build-Target([string]$name, [string[]]$sources) {
        $out = Join-Path $outDir "$name.exe"
        Write-Host "`n>> $name.exe" -ForegroundColor Yellow
        & $gxx @cxxflags @sources -o $out @ldflags
        if ($LASTEXITCODE -ne 0) { throw "build failed: $name" }
        Write-Host "   ok -> $out" -ForegroundColor Green
    }

    Build-Target 'puzzle_worker' @('main.cpp', 'kernels/opencl/opencl_solver.cpp')
    Build-Target 'test_correctness' @('test_correctness.cpp')
    Build-Target 'test_opencl_correctness' @('test_opencl_correctness.cpp', 'kernels/opencl/opencl_solver.cpp')

    Write-Host "`nbuild complete: $outDir" -ForegroundColor Green

    if ($Test) {
        Write-Host "`n=== test_correctness (CPU hash160 vs puzzle ground truth) ===" -ForegroundColor Cyan
        & (Join-Path $outDir 'test_correctness.exe')
        Write-Host "`n=== test_opencl_correctness (GPU search vs CPU) ===" -ForegroundColor Cyan
        & (Join-Path $outDir 'test_opencl_correctness.exe')
    }
}
finally {
    Pop-Location
}
