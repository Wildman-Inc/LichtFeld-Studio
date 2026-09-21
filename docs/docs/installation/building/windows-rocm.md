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

ROCm release numbers and HIP component versions differ: the tested ROCm `10.2.0` SDK reports HIP `7.16.26373` and AMD Clang `24.0.0git`. A HIP `7.x` version in the CMake summary therefore does not mean an older ROCm release was selected. Check `_rocm_sdk_core/.info/version` for the ROCm release.

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

## Hardware JPEG decoding

Windows HIP builds enable `LFS_ENABLE_ROCJPEG` by default and fetch the
[Windows rocJPEG fork](https://github.com/Yasei-no-otoko/rocJPEG) at a fixed
commit. It follows rocJPEG 1.10.0 from
[ROCm/rocm-systems](https://github.com/ROCm/rocm-systems/tree/73e42c4112d08e05170340f3fd2b5e291c2d4957/projects/rocjpeg),
including its asynchronous public API. AMF headers are fetched at a fixed commit;
the AMF decoder runtime comes from the installed AMD display driver.

Baseline JPEG images use the GPU's VCN decoder. Decoded pixels pass through
shared D3D11/D3D12 buffers into HIP for color conversion, resizing, and CHW
output. This path does not read decoded pixels back to the CPU. Windows hardware
decoding supports 4:2:0, 4:2:2, and grayscale JPEGs. Progressive, CMYK, 4:4:4,
4:4:0, unsupported dimensions, or unavailable hardware use the existing CPU
decoder. The Windows backend rejects 4:4:4/4:4:0 before entering AMF because its
BGRA decoder initialization crashed in the tested AMD driver. Encoding and
auxiliary mask/depth loading keep their existing paths.

Set `LFS_DISABLE_ROCJPEG=1` in the process environment to select CPU JPEG
decoding at runtime, or configure with `-DLFS_ENABLE_ROCJPEG=OFF` to omit this
dependency. Logs report hardware availability and hardware/CPU decode counts.

To run the loader's GPU regression suite on a supported AMD GPU:

```powershell
cmake -S . -B build-rocm10 -DBUILD_ROCJPEG_TESTS=ON
cmake --build build-rocm10 --config Release --target lichtfeld_rocjpeg_tests --parallel 32
ctest --test-dir build-rocm10 -C Release -R lichtfeld_rocjpeg_tests --output-on-failure
```

## Known constraints

- Runtime results below cover Windows 11 and Radeon 8060S (`gfx1151`) only. Other AMD architectures can be selected with `LFS_AMDGPU_ARCH`; CDNA coverage remains compile-only.
- Ordinary Win32 timeline imports, CUDA-style VMM Win32 exports, and direct imported image surfaces did not work on the tested ROCm 10.1 runtime. Windows HIP uses D3D12 fences and Vulkan-created shared buffers instead.
- Training and VkSplat use separate GPU scratch allocations on Windows HIP so each can grow independently. This uses more memory than the CUDA path's shared VMM scratch arena. UI thumbnails use host uploads; viewport images use GPU buffer copies.

## Verification status

Results recorded on 2026-09-20 cover the integration of upstream commit [c1c0f3128bbe133fbf42f803c21df0b55d6e9cbc](https://github.com/MrNeRF/LichtFeld-Studio/commit/c1c0f3128bbe133fbf42f803c21df0b55d6e9cbc). The test system ran Windows 11 on Radeon 8060S (`gfx1151`), with ROCm SDK `10.1.0a20260909`, HIP `7.16.26362`, and AMD Clang `24.0.0git`.

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

The 2026-09-21 follow-up integrates upstream [9b960dfef13dcb31068ef25ca44ef7127110f0c9](https://github.com/MrNeRF/LichtFeld-Studio/commit/9b960dfef13dcb31068ef25ca44ef7127110f0c9). Its original-JPEG shortcut is restricted to builds with an active nvImageCodec decoder so HIP does not enqueue work without a consumer. On the same system, the Release build, 49 ROCm regression tests, 68 Unicode path tests, 36 visualizer tests, and 90 format tests passed (one optional format scale simulation was skipped). Original-size JPEG training completed with both UInt8 and Float32 output; a resized GUT/PPISP/Sparsity run completed 40 steps with project save and PLY export.

The changed upstream Python suites, run with the bundled Python 3.12/native module, produced 848 passes, 2 skips, and 28 failures from existing Windows test assumptions: 25 require unavailable symlink privileges, and the remaining three assume nanosecond file timestamps, POSIX home-directory overrides, or POSIX path separators. These failures are separate from the 49 passing ROCm regression tests.

The Windows JPEG integration was subsequently verified on ROCm `10.2.0` / HIP
`7.16.26373` with the same GPU. The fetched rocJPEG commit
[`061d6c8282a076406630590271083c25c756cb71`](https://github.com/Yasei-no-otoko/rocJPEG/commit/061d6c8282a076406630590271083c25c756cb71)
passed all seven fork CTest cases, including 406 GPU conversion cases and all
13 public APIs. All seven LFS loader tests passed, covering CPU RGB comparison,
UInt8/Float32 resizing, tensor lifetime, repeated immediate/prefetch loading,
and CPU fallback for progressive/4:4:4 JPEG and PNG. The 49 ROCm regressions and
four additional package-contract cases passed. A 40-step GUT/PPISP/Sparsity run
reported 44 hardware JPEG decodes and zero CPU decodes; an original-resolution
Float32 run reported eight hardware decodes and zero CPU decodes. Both saved a
project and exported PLY files with all 12,000 and 30,000 points finite.

The same integration also passed a compatibility check with the official
`10.1.0a20260909` SDK / HIP `7.16.26362` installed in an isolated directory.
The fork was rebuilt with that SDK using 32 parallel jobs, and all seven fork
CTest cases passed. The LFS executables built with 10.2 were then run with the
10.1 runtime DLLs, matching `gfx1151` rocRAND archive, and the 10.1-built
rocJPEG DLL; this checks runtime compatibility, not a full LFS rebuild with
10.1. All seven loader tests passed, including CPU fallback. The 40-step
GUT/PPISP/Sparsity run again reported 44 hardware decodes and zero CPU decodes;
the original-resolution Float32 run reported eight and zero. Both saved a
project and exported finite PLY data (12,000 and 30,000 points respectively).
SDK artifact hashes and the running processes' loaded DLL paths were checked,
with only Windows directories on PATH. The installed 10.2 SDK was unchanged.
