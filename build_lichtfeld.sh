#!/bin/bash
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# LichtFeld-Studio Build Script for Linux
# Supports both NVIDIA CUDA and AMD ROCm backends
# Usage: ./build_lichtfeld.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
GPU_BACKEND="auto"  # auto, cuda, hip
CLEAN_BUILD=false
SKIP_VERIFICATION=false
SKIP_VCPKG=false
SKIP_LIBTORCH=false
BUILD_TESTS=false
JOBS=""

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VCPKG_PATH="${PROJECT_ROOT}/../vcpkg"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
}

print_status() {
    local name="$1"
    local status="$2"
    local info="$3"
    
    if [ "$status" = "OK" ]; then
        printf "%-35s ${GREEN}[OK]${NC}" "$name"
    elif [ "$status" = "WARN" ]; then
        printf "%-35s ${YELLOW}[WARN]${NC}" "$name"
    else
        printf "%-35s ${RED}[FAIL]${NC}" "$name"
    fi
    
    if [ -n "$info" ]; then
        echo -e " - $info"
    else
        echo ""
    fi
}

show_help() {
    cat << EOF
LichtFeld-Studio Build Script for Linux

Usage: ./build_lichtfeld.sh [options]

This script automatically:
  1. Verifies build prerequisites (compilers, GPU toolkit, CMake, Git)
  2. Sets up vcpkg
  3. Downloads LibTorch if missing
  4. Configures and builds LichtFeld-Studio

Options:
  -t, --type <Debug|Release>    Build type (default: Release)
  -g, --gpu <cuda|hip|auto>     GPU backend (default: auto-detect)
  -j, --jobs <N>                Number of parallel jobs
  -c, --clean                   Clean build directory before building
  -s, --skip-verify             Skip environment verification
  --skip-vcpkg                  Skip vcpkg setup
  --skip-libtorch               Skip LibTorch download
  --tests                       Build tests
  -h, --help                    Show this help message

Examples:
  ./build_lichtfeld.sh                          Build Release with auto-detected GPU
  ./build_lichtfeld.sh -t Debug                 Build Debug
  ./build_lichtfeld.sh -g hip                   Force ROCm/HIP backend
  ./build_lichtfeld.sh -g cuda -c               Clean and rebuild with CUDA
  ./build_lichtfeld.sh --tests                  Build with tests

GPU Backend Auto-Detection:
  - If nvidia-smi is found, CUDA is selected
  - If rocminfo is found, ROCm/HIP is selected
  - Use -g option to override auto-detection

EOF
    exit 0
}

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_BACKEND="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -s|--skip-verify)
            SKIP_VERIFICATION=true
            shift
            ;;
        --skip-vcpkg)
            SKIP_VCPKG=true
            shift
            ;;
        --skip-libtorch)
            SKIP_LIBTORCH=true
            shift
            ;;
        --tests)
            BUILD_TESTS=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

# Validate build type
if [[ ! "$BUILD_TYPE" =~ ^(Debug|Release)$ ]]; then
    echo -e "${RED}Invalid build type: $BUILD_TYPE. Must be Debug or Release.${NC}"
    exit 1
fi

# ============================================================================
# GPU Backend Detection
# ============================================================================

detect_gpu_backend() {
    if [ "$GPU_BACKEND" = "auto" ]; then
        # Check for NVIDIA first
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
            GPU_BACKEND="cuda"
            echo -e "${GREEN}Auto-detected NVIDIA GPU (CUDA)${NC}"
        # Check for AMD ROCm
        elif command -v rocminfo &> /dev/null && rocminfo &> /dev/null; then
            GPU_BACKEND="hip"
            echo -e "${GREEN}Auto-detected AMD GPU (ROCm/HIP)${NC}"
        else
            echo -e "${RED}No supported GPU detected. Please specify with -g option.${NC}"
            exit 1
        fi
    fi
}

# ============================================================================
# Environment Verification
# ============================================================================

verify_environment() {
    print_header "Verifying Build Environment"
    
    local all_passed=true
    
    # Check GCC
    echo -e "${YELLOW}[1/6] Checking C++ compiler...${NC}"
    if command -v g++ &> /dev/null; then
        local gcc_version=$(g++ --version | head -n1)
        print_status "GCC/G++" "OK" "$gcc_version"
    else
        print_status "GCC/G++" "FAIL" "Not found"
        all_passed=false
    fi
    
    # Check CMake
    echo -e "${YELLOW}[2/6] Checking CMake...${NC}"
    if command -v cmake &> /dev/null; then
        local cmake_version=$(cmake --version | head -n1 | sed 's/cmake version //')
        local cmake_major=$(echo "$cmake_version" | cut -d. -f1)
        local cmake_minor=$(echo "$cmake_version" | cut -d. -f2)
        
        if [ "$cmake_major" -ge 3 ] && [ "$cmake_minor" -ge 30 ]; then
            print_status "CMake" "OK" "v$cmake_version"
        else
            print_status "CMake" "FAIL" "v$cmake_version (need >= 3.30)"
            all_passed=false
        fi
    else
        print_status "CMake" "FAIL" "Not found"
        all_passed=false
    fi
    
    # Check Git
    echo -e "${YELLOW}[3/6] Checking Git...${NC}"
    if command -v git &> /dev/null; then
        local git_version=$(git --version | sed 's/git version //')
        print_status "Git" "OK" "v$git_version"
    else
        print_status "Git" "FAIL" "Not found"
        all_passed=false
    fi
    
    # Check GPU Toolkit
    echo -e "${YELLOW}[4/6] Checking GPU Toolkit...${NC}"
    if [ "$GPU_BACKEND" = "cuda" ]; then
        if command -v nvcc &> /dev/null; then
            local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
            print_status "CUDA Toolkit (nvcc)" "OK" "v$cuda_version"
        else
            print_status "CUDA Toolkit (nvcc)" "FAIL" "Not found"
            all_passed=false
        fi
    elif [ "$GPU_BACKEND" = "hip" ]; then
        if command -v hipcc &> /dev/null; then
            local hip_version=$(hipcc --version 2>&1 | grep "HIP version" | sed 's/.*HIP version: //')
            print_status "ROCm/HIP (hipcc)" "OK" "v$hip_version"
        else
            print_status "ROCm/HIP (hipcc)" "FAIL" "Not found"
            all_passed=false
        fi
    fi
    
    # Check Ninja (optional)
    echo -e "${YELLOW}[5/6] Checking Ninja (optional)...${NC}"
    if command -v ninja &> /dev/null; then
        local ninja_version=$(ninja --version)
        print_status "Ninja" "OK" "v$ninja_version"
    else
        print_status "Ninja" "WARN" "Not found (will use make)"
    fi
    
    # Check disk space
    echo -e "${YELLOW}[6/6] Checking disk space...${NC}"
    local free_gb=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$free_gb" -gt 30 ]; then
        print_status "Disk Space" "OK" "${free_gb}GB available"
    elif [ "$free_gb" -gt 15 ]; then
        print_status "Disk Space" "WARN" "${free_gb}GB available (may be insufficient)"
    else
        print_status "Disk Space" "FAIL" "${free_gb}GB available (need >= 15GB)"
        all_passed=false
    fi
    
    echo ""
    
    if [ "$all_passed" = false ]; then
        echo -e "${RED}Environment verification failed. Please fix the issues above.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Environment verification passed!${NC}"
    echo ""
}

# ============================================================================
# vcpkg Setup
# ============================================================================

setup_vcpkg() {
    print_header "Setting Up vcpkg"
    
    echo "vcpkg location: $VCPKG_PATH"
    echo ""
    
    if [ ! -d "$VCPKG_PATH" ]; then
        echo -e "${YELLOW}Cloning vcpkg repository...${NC}"
        git clone https://github.com/microsoft/vcpkg.git "$VCPKG_PATH"
        echo -e "${GREEN}vcpkg cloned successfully!${NC}"
    else
        echo -e "${YELLOW}vcpkg directory exists. Pulling latest changes...${NC}"
        pushd "$VCPKG_PATH" > /dev/null
        git pull || echo -e "${YELLOW}Warning: git pull failed, continuing with existing version${NC}"
        popd > /dev/null
        echo -e "${GREEN}vcpkg updated!${NC}"
    fi
    
    # Bootstrap vcpkg if needed
    if [ ! -f "$VCPKG_PATH/vcpkg" ]; then
        echo ""
        echo -e "${YELLOW}Bootstrapping vcpkg...${NC}"
        pushd "$VCPKG_PATH" > /dev/null
        ./bootstrap-vcpkg.sh -disableMetrics
        popd > /dev/null
        echo -e "${GREEN}vcpkg bootstrapped!${NC}"
    fi
    
    export VCPKG_ROOT="$VCPKG_PATH"
    echo ""
    echo -e "${CYAN}VCPKG_ROOT set to: $VCPKG_PATH${NC}"
    echo ""
}

# ============================================================================
# LibTorch Download
# ============================================================================

setup_libtorch() {
    print_header "Setting Up LibTorch"
    
    local external_dir="$PROJECT_ROOT/external"
    local libtorch_dir=""
    local libtorch_url=""
    
    mkdir -p "$external_dir"
    
    if [ "$GPU_BACKEND" = "cuda" ]; then
        libtorch_dir="$external_dir/libtorch"
        # LibTorch CUDA 12.8 URL
        libtorch_url="https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip"
    elif [ "$GPU_BACKEND" = "hip" ]; then
        libtorch_dir="$external_dir/libtorch-rocm"
        # Note: PyTorch ROCm builds may need to be obtained differently
        # For now, we'll use pip-installed torch and find its cmake directory
        echo -e "${YELLOW}For ROCm, LibTorch is typically obtained via pip install torch (ROCm version)${NC}"
        echo -e "${YELLOW}Checking for pip-installed PyTorch ROCm...${NC}"
        
        # Try to find torch cmake directory from pip installation
        local torch_cmake=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || true)
        if [ -n "$torch_cmake" ] && [ -d "$torch_cmake" ]; then
            echo -e "${GREEN}Found PyTorch CMake path: $torch_cmake${NC}"
            # Create symlink
            mkdir -p "$libtorch_dir/share/cmake"
            ln -sf "$torch_cmake" "$libtorch_dir/share/cmake/Torch" 2>/dev/null || true
            return
        else
            echo -e "${YELLOW}PyTorch ROCm not found via pip. Please install with:${NC}"
            echo "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
            echo ""
        fi
        return
    fi
    
    if [ ! -d "$libtorch_dir" ]; then
        echo ""
        echo -e "${YELLOW}Downloading LibTorch...${NC}"
        echo "This is a large download (~2.5 GB). Please wait..."
        
        local zip_file="$PROJECT_ROOT/libtorch.zip"
        
        if command -v curl &> /dev/null; then
            curl -L -o "$zip_file" "$libtorch_url" --progress-bar
        elif command -v wget &> /dev/null; then
            wget -O "$zip_file" "$libtorch_url" --show-progress
        else
            echo -e "${RED}Neither curl nor wget found. Please install one of them.${NC}"
            exit 1
        fi
        
        echo -e "${YELLOW}Extracting LibTorch...${NC}"
        unzip -q "$zip_file" -d "$external_dir"
        rm "$zip_file"
        
        echo -e "${GREEN}LibTorch installed successfully!${NC}"
    else
        echo -e "${GREEN}LibTorch already exists. Skipping download.${NC}"
    fi
    
    echo ""
}

# ============================================================================
# Build LichtFeld-Studio
# ============================================================================

build_project() {
    print_header "Building LichtFeld-Studio ($BUILD_TYPE, $GPU_BACKEND)"
    
    local build_dir="$PROJECT_ROOT/build"
    local vcpkg_toolchain="$VCPKG_PATH/scripts/buildsystems/vcpkg.cmake"
    
    # Verify vcpkg toolchain
    if [ ! -f "$vcpkg_toolchain" ]; then
        echo -e "${RED}vcpkg toolchain file not found!${NC}"
        echo "Expected: $vcpkg_toolchain"
        echo "Please run without --skip-vcpkg to set up vcpkg first."
        exit 1
    fi
    
    # Clean if requested
    if [ "$CLEAN_BUILD" = true ] && [ -d "$build_dir" ]; then
        echo -e "${YELLOW}Cleaning build directory...${NC}"
        rm -rf "$build_dir"
        echo -e "${GREEN}Build directory cleaned.${NC}"
        echo ""
    fi
    
    # Determine generator
    local generator="Unix Makefiles"
    if command -v ninja &> /dev/null; then
        generator="Ninja"
    fi
    
    # Determine job count
    if [ -z "$JOBS" ]; then
        JOBS=$(nproc)
        # Leave 2 cores free
        if [ "$JOBS" -gt 2 ]; then
            JOBS=$((JOBS - 2))
        fi
    fi
    
    # Set GPU backend flags
    local gpu_flags=""
    if [ "$GPU_BACKEND" = "cuda" ]; then
        gpu_flags="-DUSE_CUDA=ON -DUSE_HIP=OFF"
    elif [ "$GPU_BACKEND" = "hip" ]; then
        gpu_flags="-DUSE_CUDA=OFF -DUSE_HIP=ON"
    fi
    
    # Test flags
    local test_flags=""
    if [ "$BUILD_TESTS" = true ]; then
        test_flags="-DBUILD_TESTS=ON"
    fi
    
    echo "Configuration:"
    echo "  Generator: $generator"
    echo "  Build type: $BUILD_TYPE"
    echo "  GPU backend: $GPU_BACKEND"
    echo "  Jobs: $JOBS"
    echo "  Toolchain: $vcpkg_toolchain"
    echo ""
    
    # Configure
    echo -e "${YELLOW}Configuring CMake...${NC}"
    cmake -B "$build_dir" \
        -G "$generator" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_TOOLCHAIN_FILE="$vcpkg_toolchain" \
        $gpu_flags \
        $test_flags
    
    echo ""
    echo -e "${GREEN}CMake configuration successful!${NC}"
    echo ""
    
    # Build
    echo -e "${YELLOW}Building LichtFeld-Studio...${NC}"
    echo "This may take 10-30 minutes depending on your system..."
    echo ""
    
    cmake --build "$build_dir" -j "$JOBS"
    
    echo ""
    print_header "Build Successful!"
    echo "Executable location:"
    echo -e "  ${GREEN}$build_dir/LichtFeld-Studio${NC}"
    echo ""
}

# ============================================================================
# Main Execution
# ============================================================================

echo ""
print_header "LichtFeld-Studio Build Script"
echo "Project: $PROJECT_ROOT"
echo "Build type: $BUILD_TYPE"
echo ""

# Phase 1: Detect GPU backend
detect_gpu_backend

# Phase 2: Environment Verification
if [ "$SKIP_VERIFICATION" = false ]; then
    verify_environment
fi

# Phase 3: vcpkg Setup
if [ "$SKIP_VCPKG" = false ]; then
    setup_vcpkg
else
    echo -e "${YELLOW}Skipping vcpkg setup${NC}"
    if [ -d "$VCPKG_PATH" ]; then
        export VCPKG_ROOT="$VCPKG_PATH"
    else
        echo -e "${RED}Warning: vcpkg not found at $VCPKG_PATH${NC}"
    fi
    echo ""
fi

# Phase 4: LibTorch Download
if [ "$SKIP_LIBTORCH" = false ]; then
    setup_libtorch
else
    echo -e "${YELLOW}Skipping LibTorch setup${NC}"
    echo ""
fi

# Phase 5: Build
build_project

echo -e "${GREEN}All done!${NC}"
echo ""
