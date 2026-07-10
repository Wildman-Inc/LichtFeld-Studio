---
sidebar_position: 2
title: Windows ROCm and Intel Arc viewer
---

# Windows ROCm and Intel Arc viewer builds

LichtFeld Studio ROCm exposes two separate Windows products:

| Product | `LFS_PRODUCT_MODE` | `LFS_GPU_BACKEND` | Scope |
| --- | --- | --- | --- |
| AMD Studio | `STUDIO` | `HIP` | Full Studio application with ROCm/HIP training and Vulkan visualization |
| Intel Arc viewer | `VIEWER` | `NONE` | Standalone SDL3/Vulkan PLY point viewer |

`VIEWER` requires `NONE`, and `NONE` is not supported with `STUDIO`. Keep the two products in separate build directories because they use different compilers and vcpkg manifests.

## Prerequisites

Both builds require Windows x64, Visual Studio 2022 with Desktop development with C++, Git, CMake 3.30 or newer, Ninja, and a bootstrapped vcpkg checkout. Set `VCPKG_ROOT` to that checkout. Run the manual Ninja commands from an x64 Developer PowerShell for Visual Studio 2022, or initialize the same environment as shown below. The one-shot script initializes it automatically. The commands use at least 32 parallel jobs for both vcpkg and the native build.

The AMD build additionally requires a Windows ROCm/HIP SDK. The validated environment used the Python wheel layout with `_rocm_sdk_core`, `_rocm_sdk_devel`, and `_rocm_sdk_libraries_gfx1151`; `_rocm_sdk_core/include/hip/hip_version.h` reports `7.14.60850`.

The Arc viewer requires a Vulkan runtime and a Vulkan SDK containing `glslc` or `glslangValidator` for shader compilation. It does not require ROCm, CUDA, or LibTorch.

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

## AMD ROCm 7.14 Studio

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

The repository's one-shot script is an equivalent shortcut for this `gfx1151` Studio configuration. It uses `build-hip`, Ninja, and 32 build jobs:

```powershell
.\build_lichtfeld.ps1 -ProductMode STUDIO -Configuration Release -GpuBackend HIP -RocmPath $RocmRoot
```

The full Studio target includes dataset loading, HIP training, the Vulkan/VkSplat viewport, editing, export, Python, and MCP surfaces. The verified workflow is narrower than that complete feature surface; see [Verification status](#verification-status).

When external timeline semaphore interop is unavailable, the Studio viewport image path uses an asynchronous ring. Vulkan layout submissions are tracked with fence-backed tickets, while copy completion is polled through HIP events exposed under the compatibility layer's `cudaEvent*` names.

## Intel Arc standalone viewer

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

The viewer accepts one PLY path or a file dropped onto its window. It supports PLY 1.0 in ASCII and `binary_little_endian` formats, explicit RGB properties, and Gaussian splat `f_dc_0..2` SH colors. It provides orbit, pan, zoom, and point-size controls. Automatic Vulkan device selection prioritizes a compatible Intel discrete GPU (vendor ID `0x8086`), then selects the highest-ranked compatible fallback. `--gpu INDEX` bypasses automatic selection.

The install tree and generated ZIP include the executable, SPIR-V shaders, SDL3, the vcpkg Vulkan loader, redistributable MSVC runtime DLLs, documentation, and dependency license notices.

## Known constraints

- AMD validation currently covers only Radeon 8060S (`gfx1151`) with HIP `7.14.60850`. Other AMD architectures can be selected with `LFS_AMDGPU_ARCH`, but are not part of the verified matrix.
- The fence-ticket/HIP-event fallback applies to asynchronous Studio viewport image publication. It does not imply that every VkSplat transfer remains asynchronous when external timeline semaphore support is absent.
- The Arc target is a standalone Vulkan PLY point viewer. It has no training, Studio editing UI, plugins, MCP, or Studio export pipeline.
- `binary_big_endian` PLY is not supported. Scale, rotation, opacity, and higher-order SH coefficients are intentionally ignored by this point-viewer target.
- A compatible Intel discrete GPU is preferred by device selection, but no physical Intel GPU has been validated yet. The compatible fallback path was tested on AMD hardware.

## Verification status

Status as of 2026-07-10:

| Area | Status |
| --- | --- |
| AMD build and runtime | Validated on Radeon 8060S (`gfx1151`) with HIP `7.14.60850` |
| AMD training | Validated with the `truck` dataset and Sparsity + GUT + PPISP enabled |
| Studio lifecycle and viewport | VkSplat rendering plus reset, start, and stop operations worked |
| No external timeline semaphore | Async viewport fallback is implemented with Vulkan fence tickets and HIP events |
| Arc viewer build/package | Release executable, portable install tree, ZIP, and SHA-256 generated |
| Arc viewer PLY handling | Eight parser tests cover ASCII, binary, SH color, truncation, size limits, and robust finite framing |
| Arc viewer Windows CLI | `--gpu 0` loaded a real PLY through a Japanese path and filename |
| Arc viewer on Intel hardware | Not yet validated |
| Arc viewer fallback GPU | Real Gaussian PLY, SH color, resize, and orbit tested on AMD Vulkan hardware |
