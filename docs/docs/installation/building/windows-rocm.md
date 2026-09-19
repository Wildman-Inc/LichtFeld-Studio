---
sidebar_position: 2
title: LichtFeld Studio for ROCm
---

# LichtFeld Studio for ROCm builds

This guide covers Windows. For Linux ROCm and CDNA targets, use the [Linux ROCm and CDNA build guide](linux-rocm.md).

The [Wildman-Inc/LichtFeld-Studio fork](https://github.com/Wildman-Inc/LichtFeld-Studio) builds the full Studio application with ROCm/HIP training and Vulkan visualization while preserving the upstream CUDA and Linux HIP paths.

## Prerequisites

The ROCm target requires Windows x64, Visual Studio 2022 or newer with Desktop development with C++ and the Windows SDK, Git, CMake 3.30 or newer, Ninja, and a bootstrapped vcpkg checkout. Set `VCPKG_ROOT` to that checkout. Run the manual Ninja commands from an x64 Developer PowerShell, or initialize the environment as shown below. The one-shot script initializes it automatically. The commands use at least 32 parallel jobs for both vcpkg and the native build. Application inference uses native NN kernels and does not require LibTorch; optional tensor comparison tests have separate test dependencies.

The build additionally requires a Windows ROCm/HIP SDK. ROCm 10.x Python wheels use `_rocm_sdk_core`, `_rocm_sdk_devel`, and the multi-architecture `_rocm_sdk_libraries` package. Initialize the development files with `py -3.13 -m rocm_sdk init` after installing these packages in Python 3.13. The selected core's sibling development and runtime directories take precedence over other Python environments. Older per-architecture runtime wheels are used only when the multi-architecture runtime is absent.

ROCm release numbers and HIP component versions differ: the installed ROCm `10.1.0a20260909` SDK reports HIP `7.16.26362` and AMD Clang `24.0.0git`. A HIP `7.x` version in the CMake summary therefore does not mean an older ROCm release was selected. Check `_rocm_sdk_core/.info/version` for the ROCm release.

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

The hardware target is Radeon 8060S / `gfx1151`. Resolve the SDK from the intended Python interpreter explicitly, especially when `python` on PATH belongs to Conda or vcpkg. Use a separate build directory when changing SDKs:

```powershell
$RocmRoot = py -3.13 -c "import importlib.util, pathlib; spec = importlib.util.find_spec('_rocm_sdk_core'); print(pathlib.Path(spec.origin).resolve().parent if spec and spec.origin else '')"
if (-not $RocmRoot) { throw '_rocm_sdk_core was not found in Python 3.13' }

cmake -S . -B build-rocm10 -G "Ninja Multi-Config" `
  "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DLFS_GPU_BACKEND=HIP `
  "-DLFS_ROCM_PATH=$RocmRoot" `
  -DLFS_AMDGPU_ARCH=gfx1151 `
  -DLFS_VCPKG_MAX_CONCURRENCY=32

cmake --build build-rocm10 --config Release --parallel 32
.\build-rocm10\Release\LichtFeld-Studio.exe
```

The repository's one-shot script is an equivalent shortcut for this `gfx1151` LichtFeld Studio for ROCm configuration. It uses `build-hip`, Ninja Multi-Config, and 32 build jobs:

```powershell
.\build_lichtfeld.ps1 -Configuration Release -GpuBackend HIP -RocmPath $RocmRoot
```

Add `-Package` to build a portable ZIP and SHA-256 sidecar in `build-hip`. For a
manual build, configure with `-DBUILD_PORTABLE=ON`, then run
`cmake --build build-rocm10 --config Release --target package --parallel 32`.
The one-shot script additionally validates the ZIP against the selected SDK's
DLL, `.kpack`, available notice and provenance hashes. This checks package
integrity; redistribution licensing for ROCm 10.x has not been re-audited.

For a custom SDK layout, pass `LFS_ROCM_DEVEL_PATH` and `LFS_ROCM_RUNTIME_PATH` to CMake explicitly. The latter accepts directories containing runtime DLLs. Discovery never imports development headers from an unrelated Python environment when an SDK root has been selected.

ROCm 10.x separates rocRAND device kernels into `.kpack` archives. The build stages only the archives needed by the selected `LFS_AMDGPU_ARCH` targets, one directory above the runtime DLLs. The Multi-Config layout keeps these files inside the build directory. When switching an existing Ninja build, rerun configuration with `--fresh -G "Ninja Multi-Config"`; the dependency installation is retained.

LichtFeld Studio for ROCm includes dataset loading, HIP training, the Vulkan/VkSplat viewport, editing, export, Python, and MCP surfaces. The verified workflow is narrower than that complete feature surface; see [Verification status](#verification-status).

Windows HIP synchronizes Vulkan and HIP through a D3D12 shared fence, imported as a Vulkan timeline semaphore and a HIP D3D12 fence. Vulkan allocates shared model buffers that HIP maps directly. HIP writes RGBA8 and R32F scene output into shared linear buffers; Vulkan copies those buffers into images for presentation. The selected GPU and driver must support these fence and buffer imports.

## Known constraints

- Runtime results below cover Windows 11 and Radeon 8060S (`gfx1151`) only. Other AMD architectures can be selected with `LFS_AMDGPU_ARCH`; CDNA coverage remains compile-only.
- Ordinary Win32 timeline imports, CUDA-style VMM Win32 exports, and direct imported image surfaces did not work on the tested ROCm 10.1 runtime. Windows HIP uses D3D12 fences and Vulkan-created shared buffers instead.
- Training and VkSplat use separate GPU scratch allocations on Windows HIP so each can grow independently. This uses more memory than the CUDA path's shared VMM scratch arena. UI thumbnails use host uploads; viewport images use GPU buffer copies.

## Verification status

Results recorded on 2026-09-20 cover the integration of upstream commit [0e13e8ff469dee10a69814e3e512d6451e2ada98](https://github.com/MrNeRF/LichtFeld-Studio/commit/0e13e8ff469dee10a69814e3e512d6451e2ada98). The test system ran Windows 11 on Radeon 8060S (`gfx1151`), with ROCm SDK `10.1.0a20260909`, HIP `7.16.26362`, and AMD Clang `24.0.0git`.

| Area | Status |
| --- | --- |
| Full application build | Release portable build completed with 32 parallel jobs |
| Focused regression tests | 49 pytest tests passed |
| Native NN kernels | GPU attention and convolution numerical parity checks passed |
| rocRAND device archives | Generation of 1,024 finite uniform values passed with the selected `.kpack` archive |
| Headless training and export | `truck` completed 40 steps with GUT, PPISP, and Sparsity; project and PLY saved, with all 62,988 exported points finite |
| GPU synchronization | Vulkan-to-HIP and HIP-to-Vulkan D3D12 shared-fence probes passed |
| Image conversion kernels | 72 GPU cases passed for RGBA8/R32F, HWC/CHW, channel counts, flips, odd dimensions, padded rows, and invalid pitches |
| GUI lifecycle and viewport | Project hydration, 3DGS/3DGUT display, pause/resume, dataset reload, graceful stop, and selection/clear passed. GUI training completed 1,000 steps (loss 0.114351); all 173,609 PLY points were finite |
| Portable ZIP | SDK artifact hashes and ZIP policy passed; extracted executable started and completed GUT/PPISP/Sparsity training and export with only Windows directories on PATH |

CUDA and Linux HIP paths are retained. Linux and CDNA compile coverage is tracked separately in the [Linux ROCm and CDNA build guide](linux-rocm.md); no physical CDNA runtime or training result is claimed here.
