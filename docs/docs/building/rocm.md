---
sidebar_position: 20
---

# ROCm / HIP Build (Linux)

This page documents the Linux-first ROCm build path for LichtFeld Studio.

## Supported Scope

- OS: Ubuntu 24.04+
- ROCm: 7.x (tested target: ROCm 7.2+)
- GPU: AMD Instinct MI300X and newer (`gfx942` minimum target in this guide)
- Backend switch: `LFS_GPU_BACKEND=HIP`

## Prerequisites

1. Install ROCm 7.x, including HIP toolchain (`hipcc`) and runtime.
2. Ensure your GPU is visible:

```bash
rocminfo | grep -E "Name:|gfx"
hipcc --version
```

3. Ensure regular project prerequisites are available (`cmake`, `ninja`, compiler toolchain, vcpkg deps).

## Configure And Build

```bash
cmake -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLFS_GPU_BACKEND=HIP \
  -DLFS_AMDGPU_ARCH=gfx942

cmake --build build-hip -j
```

### Architecture Selection

- Fixed arch (recommended for deployment):
  - `-DLFS_AMDGPU_ARCH=gfx942`
- Auto detect from `rocminfo`:
  - `-DLFS_AMDGPU_ARCH=auto`
  - Falls back to `gfx942` when detection is unavailable.

## Runtime Smoke Command

Use a short headless run to verify training loop startup:

```bash
./build-hip/LichtFeld-Studio \
  -d <dataset_path> \
  -o <output_path> \
  --train --iter 5 --headless
```

Expected behavior for smoke: process starts, initializes HIP backend, and runs first iterations without immediate crash.

## HIPIFY Helper

A helper script is available to regenerate HIP-translated source snapshots for training/rasterization kernels:

```bash
./scripts/hipify_rocm.sh
```

Output is written to `build/hipify/` by default.

## HIP-Specific CMake Options

- `LFS_GPU_BACKEND`:
  - `CUDA` (default)
  - `HIP`
- `LFS_AMDGPU_ARCH`:
  - Example: `gfx942`
  - `auto` for `rocminfo`-based detection
- `LFS_ENABLE_HIP_GL_INTEROP`:
  - `ON` by default for HIP builds
  - Auto-disabled when compile-check fails
- `LFS_ENABLE_NVIMAGECODEC`:
  - Forced `OFF` for HIP builds

## Notes

- `nvImageCodec` is NVIDIA-only and is automatically disabled in HIP builds.
- If HIP OpenGL interop is unavailable on your stack, build with:

```bash
-DLFS_ENABLE_HIP_GL_INTEROP=OFF
```

The renderer/training path will use fallback upload paths.
