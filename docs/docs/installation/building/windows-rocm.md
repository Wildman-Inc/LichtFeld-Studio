---
sidebar_position: 2
title: LichtFeld Studio for ROCm and LichtFeld Arc Viewer (Experimental)
---

# LichtFeld Studio for ROCm and LichtFeld Arc Viewer (Experimental) builds

This guide covers Windows. For Linux ROCm and CDNA targets, use the [Linux ROCm and CDNA build guide](linux-rocm.md).

The [Wildman-Inc/LichtFeld-Studio fork](https://github.com/Wildman-Inc/LichtFeld-Studio) publishes the ROCm Studio product plus a separate experimental viewer:

| Product | `LFS_PRODUCT_MODE` | `LFS_GPU_BACKEND` | Scope |
| --- | --- | --- | --- |
| LichtFeld Studio for ROCm | `STUDIO` | `HIP` | Full Studio application with ROCm/HIP training and Vulkan visualization |
| LichtFeld Arc Viewer (Experimental) | `VIEWER` | `NONE` | Standalone SDL3/Vulkan PLY point viewer; no training backend |

`VIEWER` requires `NONE`, and `NONE` is not supported with `STUDIO`. Keep the two targets in separate build directories because they use different compilers, backends, and vcpkg configurations.

## Prerequisites

Both targets require Windows x64, Visual Studio 2022 with Desktop development with C++, Git, CMake 3.30 or newer, Ninja, and a bootstrapped vcpkg checkout. Set `VCPKG_ROOT` to that checkout. Run the manual Ninja commands from an x64 Developer PowerShell for Visual Studio 2022, or initialize the same environment as shown below. The one-shot script initializes it automatically. The commands use at least 32 parallel jobs for both vcpkg and the native build.

The AMD build additionally requires a Windows ROCm/HIP SDK. The validated environment used the Python wheel layout with `_rocm_sdk_core`, `_rocm_sdk_devel`, and `_rocm_sdk_libraries_gfx1151`; `_rocm_sdk_core/include/hip/hip_version.h` reports `7.14.60850`.

LichtFeld Arc Viewer (Experimental) requires a Vulkan runtime and a Vulkan SDK containing `glslc` or `glslangValidator` for shader compilation. It does not require ROCm, CUDA, or LibTorch and contains no training backend.

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

The validated target is Radeon 8060S / `gfx1151`. Resolve the installed wheel root and configure the product explicitly as `STUDIO+HIP`:

```powershell
$RocmRoot = python -c "import importlib.util, pathlib; spec = importlib.util.find_spec('_rocm_sdk_core'); print(pathlib.Path(spec.origin).resolve().parent if spec and spec.origin else '')"
if (-not $RocmRoot) { throw '_rocm_sdk_core was not found in the active Python environment' }

cmake -S . -B build-rocm714 -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=Release `
  -DLFS_PRODUCT_MODE=STUDIO `
  -DLFS_GPU_BACKEND=HIP `
  "-DLFS_ROCM_PATH=$RocmRoot" `
  -DLFS_AMDGPU_ARCH=gfx1151 `
  -DLFS_VCPKG_MAX_CONCURRENCY=32

cmake --build build-rocm714 --config Release --parallel 32
.\build-rocm714\LichtFeld-Studio.exe
```

The repository's one-shot script is an equivalent shortcut for this `gfx1151` LichtFeld Studio for ROCm configuration. It uses `build-hip`, Ninja, and 32 build jobs:

```powershell
.\build_lichtfeld.ps1 -ProductMode STUDIO -Configuration Release -GpuBackend HIP -RocmPath $RocmRoot
```

LichtFeld Studio for ROCm includes dataset loading, HIP training, the Vulkan/VkSplat viewport, editing, export, Python, and MCP surfaces. The verified workflow is narrower than that complete feature surface; see [Verification status](#verification-status).

When external-memory scene-image interop succeeds but external timeline semaphores are unavailable, the viewport handoff uses an asynchronous external-memory ring. Vulkan layout submissions are polled through fence-backed tickets, while copy completion is recorded and polled through `cudaEvent*`; in HIP builds, the compatibility layer maps those types and calls to `hipEvent*`. Reset and teardown may synchronize an outstanding event before releasing an interop target. If an external scene image cannot be used, the scene uploader instead copies through mapped CPU/Vulkan staging memory.

## LichtFeld Arc Viewer (Experimental)

Configure the minimal product explicitly as `VIEWER+NONE`:

```powershell
cmake -S . -B build-arc-viewer -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=Release `
  -DLFS_PRODUCT_MODE=VIEWER `
  -DLFS_GPU_BACKEND=NONE `
  -DLFS_VCPKG_MAX_CONCURRENCY=32

cmake --build build-arc-viewer --config Release --parallel 32
.\build-arc-viewer\LichtFeld-Studio-Arc-Viewer.exe --help
.\build-arc-viewer\LichtFeld-Studio-Arc-Viewer.exe C:\data\scene.ply
.\build-arc-viewer\LichtFeld-Studio-Arc-Viewer.exe --gpu 0 C:\data\scene.ply

cmake --install build-arc-viewer --prefix build-arc-viewer\portable
cmake --build build-arc-viewer --target package --parallel 32
```

The one-shot script selects `NONE`, Ninja, and `build-arc-viewer` automatically:

```powershell
.\build_lichtfeld.ps1 -ProductMode VIEWER -Configuration Release
```

LichtFeld Arc Viewer (Experimental) accepts one PLY path or a file dropped onto its window. It supports PLY 1.0 in ASCII and `binary_little_endian` formats, explicit RGB properties, and Gaussian splat `f_dc_0..2` SH colors. It provides orbit, pan, zoom, and point-size controls. Automatic Vulkan device selection prioritizes a compatible Intel discrete GPU (vendor ID `0x8086`), then selects the highest-ranked compatible fallback. `--gpu INDEX` bypasses automatic selection.

The install tree and generated ZIP include the executable, SPIR-V shaders, SDL3, the vcpkg Vulkan loader, redistributable MSVC runtime DLLs, documentation, and the repository license inventory. Dependency license and notice files are included when they are discovered in the configured dependency installations.

## Known constraints

- AMD validation currently covers only Radeon 8060S (`gfx1151`) with HIP `7.14.60850`. Other AMD architectures can be selected with `LFS_AMDGPU_ARCH`, but are not part of the verified matrix.
- The Vulkan-fence/HIP-event synchronization fallback applies only when external-memory scene-image interop succeeds. CPU/Vulkan staging remains available when that image path is unavailable, and individual VkSplat transfers may still synchronize without external timeline semaphore support.
- LichtFeld Arc Viewer (Experimental) is a standalone Vulkan PLY point viewer. It is not a ROCm training target and has no training, Studio editing UI, plugins, MCP, or Studio export pipeline.
- `binary_big_endian` PLY is not supported. Scale, rotation, opacity, and higher-order SH coefficients are intentionally ignored by this point-viewer target.
- No physical Intel Arc GPU has been validated. Only the compatible AMD Vulkan fallback has been validated.

## Verification status

Status as of 2026-07-11:

| Area | Status |
| --- | --- |
| AMD build and runtime | Validated on Radeon 8060S (`gfx1151`) with HIP `7.14.60850` |
| AMD training | Validated with the `truck` dataset and Sparsity + GUT + PPISP enabled |
| Studio lifecycle and viewport | VkSplat rendering plus reset, start, and stop operations worked |
| No external timeline semaphore | Fence/HIP-event synchronization is implemented when external-memory image interop succeeds; CPU/Vulkan staging remains available otherwise |
| LichtFeld Arc Viewer (Experimental) build/package | Release executable, portable install tree, ZIP, and SHA-256 generated |
| LichtFeld Arc Viewer (Experimental) PLY handling | Eight parser tests cover ASCII, binary, SH color, truncation, size limits, and robust finite framing |
| LichtFeld Arc Viewer (Experimental) Windows CLI | `--gpu 0` loaded a real PLY through a Japanese path and filename |
| LichtFeld Arc Viewer (Experimental) on Intel hardware | Not yet validated |
| LichtFeld Arc Viewer (Experimental) fallback GPU | Real Gaussian PLY, SH color, resize, and orbit tested only on AMD Vulkan hardware |

Linux and CDNA compile coverage is tracked separately in the [Linux ROCm and CDNA build guide](linux-rocm.md).
