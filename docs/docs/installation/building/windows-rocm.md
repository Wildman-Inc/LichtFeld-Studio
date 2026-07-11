---
sidebar_position: 2
title: LichtFeld Studio for ROCm
---

# LichtFeld Studio for ROCm builds

This guide covers Windows. For Linux ROCm and CDNA targets, use the [Linux ROCm and CDNA build guide](linux-rocm.md).

The [Wildman-Inc/LichtFeld-Studio fork](https://github.com/Wildman-Inc/LichtFeld-Studio) publishes the full Studio application with ROCm/HIP training and Vulkan visualization while preserving the upstream CUDA backend.

## Prerequisites

The ROCm target requires Windows x64, Visual Studio 2022 with Desktop development with C++, Git, CMake 3.30 or newer, Ninja, and a bootstrapped vcpkg checkout. Set `VCPKG_ROOT` to that checkout. Run the manual Ninja commands from an x64 Developer PowerShell for Visual Studio 2022, or initialize the same environment as shown below. The one-shot script initializes it automatically. The commands use at least 32 parallel jobs for both vcpkg and the native build.

The build additionally requires a Windows ROCm/HIP SDK. The validated environment used the Python wheel layout with `_rocm_sdk_core`, `_rocm_sdk_devel`, and `_rocm_sdk_libraries_gfx1151`; `_rocm_sdk_core/include/hip/hip_version.h` reports `7.14.60850`.

Initialize the shared prerequisites from PowerShell:

```powershell
Set-Location C:\src\LichtFeld-Studio
$env:VCPKG_ROOT = 'C:\src\vcpkg'
$env:VCPKG_MAX_CONCURRENCY = '32'

$VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$VsPath = & $VsWhere -latest -products * `
  -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
  -property installationPath
Import-Module (Join-Path $VsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation `
  -DevCmdArguments '-arch=x64 -host_arch=x64'

git submodule update --init --recursive
```

## LichtFeld Studio for ROCm

The validated target is Radeon 8060S / `gfx1151`. Resolve the installed wheel root and configure the HIP backend:

```powershell
$RocmRoot = python -c "import importlib.util, pathlib; spec = importlib.util.find_spec('_rocm_sdk_core'); print(pathlib.Path(spec.origin).resolve().parent if spec and spec.origin else '')"
if (-not $RocmRoot) { throw '_rocm_sdk_core was not found in the active Python environment' }

cmake -S . -B build-rocm714 -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=Release `
  -DLFS_GPU_BACKEND=HIP `
  "-DLFS_ROCM_PATH=$RocmRoot" `
  -DLFS_AMDGPU_ARCH=gfx1151 `
  -DLFS_VCPKG_MAX_CONCURRENCY=32

cmake --build build-rocm714 --config Release --parallel 32
.\build-rocm714\LichtFeld-Studio.exe
```

The repository's one-shot script is an equivalent shortcut for this `gfx1151` LichtFeld Studio for ROCm configuration. It uses `build-hip`, Ninja, and 32 build jobs:

```powershell
.\build_lichtfeld.ps1 -Configuration Release -GpuBackend HIP -RocmPath $RocmRoot
```

LichtFeld Studio for ROCm includes dataset loading, HIP training, the Vulkan/VkSplat viewport, editing, export, Python, and MCP surfaces. The verified workflow is narrower than that complete feature surface; see [Verification status](#verification-status).

When external-memory scene-image interop succeeds but external timeline semaphores are unavailable, the viewport handoff uses an asynchronous external-memory ring. Vulkan layout submissions are polled through fence-backed tickets, while copy completion is recorded and polled through `cudaEvent*`; in HIP builds, the compatibility layer maps those types and calls to `hipEvent*`. Reset and teardown may synchronize an outstanding event before releasing an interop target. If an external scene image cannot be used, the scene uploader instead copies through mapped CPU/Vulkan staging memory.

## Known constraints

- AMD validation currently covers only Radeon 8060S (`gfx1151`) with HIP `7.14.60850`. Other AMD architectures can be selected with `LFS_AMDGPU_ARCH`, but are not part of the verified matrix.
- The Vulkan-fence/HIP-event synchronization fallback applies only when external-memory scene-image interop succeeds. CPU/Vulkan staging remains available when that image path is unavailable, and individual VkSplat transfers may still synchronize without external timeline semaphore support.

## Verification status

Status as of 2026-07-11:

| Area | Status |
| --- | --- |
| AMD build and runtime | Validated on Radeon 8060S (`gfx1151`) with HIP `7.14.60850` |
| AMD training | Validated with the `truck` dataset and Sparsity + GUT + PPISP enabled |
| Studio lifecycle and viewport | VkSplat rendering plus reset, start, and stop operations worked |
| No external timeline semaphore | Fence/HIP-event synchronization is implemented when external-memory image interop succeeds; CPU/Vulkan staging remains available otherwise |

Linux and CDNA compile coverage is tracked separately in the [Linux ROCm and CDNA build guide](linux-rocm.md).
