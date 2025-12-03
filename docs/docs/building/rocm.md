# ROCm/HIP Support for LichtFeld-Studio

LichtFeld-Studio now supports both NVIDIA CUDA and AMD ROCm/HIP backends for GPU computation.

> ⚠️ **Windows ROCm Support**: Experimental Windows ROCm support is now available for:
> - **AMD RX 7900 series (gfx1100)** - RDNA 3
> - **AMD RX 9070/9060 series (gfx1200/gfx1201)** - RDNA 4
> 
> Requires [AMD HIP SDK 6.x](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) and 
> [PyTorch on Windows driver](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-1-1.html).
> 
> See [Windows ROCm Build](#windows-rocm-build-experimental) for details.

## Supported Hardware

### AMD GPUs (ROCm 7.x)

| Platform | GPU Series | Architecture | Status |
|----------|-----------|--------------|--------|
| **Linux** | MI200 series | gfx90a | ✅ Fully Supported |
| **Linux** | MI300X | gfx940 | ✅ Fully Supported |
| **Linux** | MI300A | gfx942 | ✅ Fully Supported |
| **Linux** | RX 7900 series | gfx1100 | ✅ Fully Supported |
| **Linux** | RX 7600/7700 | gfx1101/1102 | ✅ Fully Supported |
| **Linux** | RX 6800/6900 | gfx1030 | ✅ Fully Supported |
| **Windows** | RX 9070/9060 | gfx1200/1201 | 🧪 Experimental |
| **Windows** | RX 7900 series | gfx1100 | 🧪 Experimental |
| **Windows** | RX 6000 series | gfx1030 | ❌ Not Supported |

## Prerequisites

### Linux (Recommended)

1. **ROCm 7.0 or later**
   ```bash
   # Ubuntu 24.04
   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_6.0.60000-1_all.deb
   sudo apt install ./amdgpu-install_6.0.60000-1_all.deb
   sudo amdgpu-install --usecase=rocm,hip
   ```

2. **ROCm Development Libraries**
   ```bash
   sudo apt install rocm-dev hipblas-dev hiprand-dev hipcub-dev rocprim-dev rocthrust-dev
   ```

3. **PyTorch ROCm**
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   ```

4. **Other dependencies** (same as CUDA build)
   ```bash
   sudo apt install cmake ninja-build git
   ```

### Windows (Experimental)

> ⚠️ **Windows ROCm requires specific hardware and drivers.**

1. **Supported GPU Required**
   - AMD RX 9070 XT, RX 9070, RX 9060 XT (RDNA 4)
   - AMD RX 7900 XTX, RX 7900 XT, PRO W7900 (RDNA 3)
   - Ryzen AI with Radeon 890M/880M (only for inference)

2. **AMD PyTorch on Windows Driver**
   - Download from: [AMD PyTorch on Windows 7.1.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-1-1.html)
   - Required driver version: **25.20.01.17** or later

3. **AMD HIP SDK**
   - Download from: [AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
   - Set `HIP_PATH` environment variable to installation directory

4. **Visual Studio 2022** with "Desktop development with C++"

5. **CMake 3.30+** and **Ninja** (recommended)

## Building with ROCm

### Linux Build (Using Script)

```bash
# Auto-detect GPU (CUDA or ROCm)
./build_lichtfeld.sh

# Force ROCm/HIP backend
./build_lichtfeld.sh -g hip

# Debug build with ROCm
./build_lichtfeld.sh -g hip -t Debug

# Clean build
./build_lichtfeld.sh -g hip -c
```

### Windows ROCm Build (Experimental)

```powershell
# Auto-detect GPU backend
.\build_lichtfeld.ps1

# Force HIP/ROCm backend
.\build_lichtfeld.ps1 -GpuBackend HIP

# Debug build with HIP
.\build_lichtfeld.ps1 -GpuBackend HIP -Configuration Debug

# Clean and rebuild
.\build_lichtfeld.ps1 -GpuBackend HIP -Clean
```

### Manual CMake Build (Linux)

```bash
# Create build directory
mkdir build && cd build

# Configure with ROCm/HIP
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF \
    -DUSE_HIP=ON \
    -DHIP_ARCHITECTURES="gfx90a;gfx1100" \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build . -j$(nproc)
```

### Manual CMake Build (Windows)

```powershell
# Set HIP SDK path
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\6.2"

# Configure with HIP
cmake -B build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DUSE_CUDA=OFF `
    -DUSE_HIP=ON `
    -DHIP_ARCHITECTURES="gfx1100;gfx1200;gfx1201" `
    -DCMAKE_CXX_COMPILER="$env:HIP_PATH\bin\clang++.exe" `
    -DCMAKE_C_COMPILER="$env:HIP_PATH\bin\clang.exe" `
    -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"

# Build
cmake --build build -j
```

### CMake Options for ROCm

| Option | Description | Default |
|--------|-------------|---------|
| `USE_HIP` | Enable ROCm/HIP backend | OFF |
| `USE_CUDA` | Enable CUDA backend | ON |
| `HIP_ARCHITECTURES` | Target GPU architectures | gfx90a;gfx940;gfx942;gfx1100;gfx1101 |
| `HIP_SDK_PATH` | Path to HIP SDK (Windows) | Auto-detected |

## Docker Support

### Building the ROCm Container

```bash
cd docker
docker build -f Dockerfile.rocm -t lichtfeld-studio-rocm ..
```

### Running with Docker Compose

```bash
# Start container
docker-compose -f docker/docker-compose.rocm.yml up -d

# Enter container
docker exec -it lichtfeld-studio-rocm bash

# Build inside container
./build_lichtfeld.sh -g hip
```

### GPU Access in Docker

For AMD GPU access in Docker, ensure:
1. The container has access to `/dev/kfd` and `/dev/dri`
2. User is in `video` and `render` groups
3. ROCm drivers are installed on the host

## Architecture Selection

Specify target architectures based on your GPU:

```cmake
# MI200 series only
-DHIP_ARCHITECTURES="gfx90a"

# MI300 series
-DHIP_ARCHITECTURES="gfx940;gfx942"

# Consumer RDNA 3
-DHIP_ARCHITECTURES="gfx1100;gfx1101"

# Multiple architectures (larger binary)
-DHIP_ARCHITECTURES="gfx90a;gfx940;gfx1100"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ROCM_PATH` | ROCm installation path (default: /opt/rocm) |
| `HIP_PATH` | HIP installation path (default: /opt/rocm) |
| `HSA_OVERRIDE_GFX_VERSION` | Override GPU detection for compatibility |
| `HIP_VISIBLE_DEVICES` | Select specific GPU(s) to use |

## Troubleshooting

### GPU Not Detected

```bash
# Check ROCm installation
rocminfo

# Check HIP installation
hipconfig --full

# List available GPUs
rocm-smi
```

### Permission Issues

```bash
# Add user to required groups
sudo usermod -aG video $USER
sudo usermod -aG render $USER
# Log out and back in
```

### Compatibility Mode

For newer GPUs not fully supported:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RDNA 3
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RDNA 2
```

### PyTorch Issues

```bash
# Verify PyTorch ROCm
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.version.hip)"
```

## Performance Notes

1. **Warp Size**: AMD GPUs use 64-wide wavefronts (vs 32 for NVIDIA), except RDNA which uses 32.
2. **Memory**: MI300X offers 192GB HBM3, MI200 offers 128GB HBM2e.
3. **Shared Memory**: Called LDS (Local Data Share) on AMD GPUs.

## Known Limitations

1. CUDA-OpenGL interop has limited support on ROCm (hip_gl_interop.h)
2. Some CUDA intrinsics may need manual porting
3. Profiling tools differ (rocprof vs nvprof)
4. **Windows ROCm is experimental** - Requires specific hardware (gfx1100+) and drivers

## Windows Limitations

### Supported vs Unsupported GPUs

| Windows Support | GPU Architecture | Example GPUs |
|-----------------|-----------------|--------------|
| ✅ Supported | gfx1200, gfx1201 | RX 9070 XT, RX 9060 XT (RDNA 4) |
| ✅ Supported | gfx1100 | RX 7900 XTX, PRO W7900 (RDNA 3) |
| ❌ Not Supported | gfx1030 | RX 6900 XT, RX 6800 XT (RDNA 2) |
| ❌ Not Supported | gfx90a, gfx940 | MI200, MI300 (Datacenter) |

### Why Some GPUs Aren't Supported on Windows

AMD's PyTorch on Windows release (7.1.1) only includes support for:
- RDNA 4 (Radeon 9000 series)
- RDNA 3 (Radeon 7900 series)

RDNA 2 and datacenter GPUs require Linux due to different driver architectures.

### Options for Unsupported Windows GPUs

If your AMD GPU is not supported on Windows:

1. **Linux Dual Boot (Recommended)**
   - Install Ubuntu 22.04/24.04 alongside Windows
   - Best performance and full ROCm support
   - [Ubuntu Dual Boot Guide](https://help.ubuntu.com/community/WindowsDualBoot)

2. **Dedicated Linux Machine**
   - Use a separate PC with Linux for GPU workloads
   - Access via SSH/remote desktop

3. **Linux VM with GPU Passthrough**
   - Requires compatible motherboard (IOMMU support)
   - Complex setup but viable option
   - [VFIO/GPU Passthrough Guide](https://wiki.archlinux.org/title/PCI_passthrough_via_OVMF)

### Why Not WSL2?

Unlike NVIDIA GPUs, AMD GPUs do **not** support GPU passthrough in WSL2. The Windows AMD driver does not expose the GPU to WSL2 for compute workloads. Only NVIDIA provides this capability via their special WSL2 CUDA drivers.

## Contributing

When contributing GPU code, please:
1. Use the `hip_compat.h` header for portable code
2. Test on both CUDA and ROCm if possible
3. Use architecture-agnostic code patterns
4. Document any backend-specific workarounds
