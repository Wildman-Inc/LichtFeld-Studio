---
sidebar_position: 3
title: Linux ROCm and CDNA
---

# Linux ROCm and CDNA builds

LichtFeld Studio for ROCm supports Linux source builds through CMake's native HIP language. The build accepts one or more AMDGPU targets, links the Linux `hip::device` and hipRAND package targets, and keeps CUDA's 32-lane logical warp behavior on wave64 CDNA devices.

The Linux path is source-only. The repository does not currently publish a Linux binary archive. CI compile coverage is described in [Verification status](#verification-status); a complete application run and training session still need validation on physical Linux CDNA hardware.

## Prerequisites

- A supported 64-bit Linux distribution and AMD GPU from the ROCm compatibility matrix
- A complete ROCm development installation containing HIP, hipRAND, and their CMake package files
- Git, CMake 3.30 or newer, Ninja, a C++ toolchain, Python development headers, and Vulkan development/runtime support
- A bootstrapped vcpkg checkout

Follow AMD's current [ROCm Linux installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-ubuntu.html) and [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) for host and GPU requirements. Verify the development installation before configuring LichtFeld Studio:

```bash
/opt/rocm/bin/hipconfig --full
/opt/rocm/bin/hipcc --version
cmake --find-package -DNAME=hip -DCOMPILER_ID=Clang -DLANGUAGE=HIP -DMODE=EXIST
cmake --find-package -DNAME=hiprand -DCOMPILER_ID=Clang -DLANGUAGE=CXX -DMODE=EXIST
```

## Configure and build

Set the vcpkg root, initialize submodules, and put the selected ROCm installation on `PATH` and `CMAKE_PREFIX_PATH`:

```bash
export VCPKG_ROOT="$HOME/src/vcpkg"
export VCPKG_MAX_CONCURRENCY=32
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export CMAKE_PREFIX_PATH="$ROCM_PATH${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

git submodule update --init --recursive
```

On a local GPU host, omitting `LFS_AMDGPU_ARCH` selects `native`. For reproducible builds, set the architecture explicitly. This example targets an MI300-class `gfx942` device:

```bash
cmake -S . -B build-rocm -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DLFS_PRODUCT_MODE=STUDIO \
  -DLFS_GPU_BACKEND=HIP \
  -DLFS_AMDGPU_ARCH=gfx942 \
  -DLFS_VCPKG_MAX_CONCURRENCY=32

cmake --build build-rocm --config Release --parallel 32
./build-rocm/LichtFeld-Studio
```

Common CDNA examples are:

| GPU family | Typical target |
| --- | --- |
| Instinct MI200 | `gfx90a` |
| Instinct MI300 | `gfx940`, `gfx941`, or `gfx942`, depending on the exact GPU |
| Instinct MI350 | `gfx950` |

Confirm the exact target with `rocminfo`; do not infer it from the marketing name alone. A multi-architecture build accepts a semicolon- or comma-separated list:

```bash
cmake -S . -B build-rocm-multi -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DLFS_PRODUCT_MODE=STUDIO \
  -DLFS_GPU_BACKEND=HIP \
  '-DLFS_AMDGPU_ARCH=gfx90a;gfx942;gfx950' \
  -DLFS_VCPKG_MAX_CONCURRENCY=32

cmake --build build-rocm-multi --config Release --parallel 32
```

Use a separate build directory for each materially different ROCm SDK or architecture set. CMake caches the HIP compiler and target list.

## Verification status

Status as of 2026-07-11:

| Area | Status |
| --- | --- |
| Windows Radeon runtime and training | Validated separately; see the [Windows guide](windows-rocm.md) |
| CDNA source compatibility | CUDA-style 32-lane reductions use explicit shuffle widths so they preserve their logical warp on wave64 hardware |
| Local compiler smoke | On WSL Ubuntu 24.04 with ROCm 7.2, the core and training warp/block reduction contracts compile for `gfx90a`, `gfx942`, and `gfx950`; representative kernels also compile for `gfx942` with HIP Clang |
| Linux CI | The `Linux ROCm CDNA Compile` workflow uses AMD's ROCm development container and compiles the wave-reduction smoke for `gfx90a`, `gfx942`, and `gfx950` |
| Full Linux application build | Not yet validated in CI or on a physical Linux host |
| Linux CDNA runtime and training | Not yet validated on physical hardware |

The CI smoke is a compile contract, not a claim of complete runtime support. Before publishing a Linux binary, validate dataset import, training progress, reset/stop lifecycle, Vulkan rendering, and the selected advanced training options on the target driver and GPU.
