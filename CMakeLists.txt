# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# windows/linux platform independent
set(RAW_VCPKG_PATH $ENV{VCPKG_ROOT})
file(TO_CMAKE_PATH "${RAW_VCPKG_PATH}" VCPKG_ROOT)

if ("${CMAKE_TOOLCHAIN_FILE}" STREQUAL "")
    message(STATUS "user didn't specify CMAKE_TOOLCHAIN_FILE manually, so let's use default -> ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()

cmake_minimum_required(VERSION 3.30)

# ============================================================================
# GPU Backend Selection
# ============================================================================
# Options: CUDA (NVIDIA), HIP (AMD ROCm)
option(USE_CUDA "Build with NVIDIA CUDA support" ON)
option(USE_HIP "Build with AMD ROCm/HIP support" OFF)

# Validate GPU backend selection
if(USE_CUDA AND USE_HIP)
    message(FATAL_ERROR "Cannot enable both CUDA and HIP backends. Please select only one.")
endif()

if(NOT USE_CUDA AND NOT USE_HIP)
    message(STATUS "No GPU backend specified. Defaulting to CUDA.")
    set(USE_CUDA ON)
endif()

# Set up languages based on selected backend
# Note: On Windows, CMake HIP language is not supported. We use CXX with hipcc as compiler.
if(USE_HIP)
    if(WIN32)
        # Windows: Use CXX language with HIP SDK's clang++
        # Note: For HIP on Windows, user should invoke CMake with:
        #   -DCMAKE_CXX_COMPILER=<HIP_PATH>/lib/llvm/bin/clang++.exe
        # Or set HIP_PATH environment variable and we'll configure it
        # Check if user already specified clang++ as the compiler
        if(CMAKE_CXX_COMPILER)
            get_filename_component(_cxx_compiler_name "${CMAKE_CXX_COMPILER}" NAME_WE)
        else()
            set(_cxx_compiler_name "")
        endif()
        
        if(NOT _cxx_compiler_name MATCHES "clang")
            # Check if HIP SDK clang is available
            # HIP SDK 6.4+ places clang in lib/llvm/bin
            if(DEFINED ENV{HIP_PATH})
                set(_hip_clang "$ENV{HIP_PATH}/lib/llvm/bin/clang++.exe")
                if(NOT EXISTS "${_hip_clang}")
                    # Fallback to bin directory for older SDK versions
                    set(_hip_clang "$ENV{HIP_PATH}/bin/clang++.exe")
                endif()
            elseif(DEFINED HIP_SDK_PATH)
                set(_hip_clang "${HIP_SDK_PATH}/lib/llvm/bin/clang++.exe")
                if(NOT EXISTS "${_hip_clang}")
                    set(_hip_clang "${HIP_SDK_PATH}/bin/clang++.exe")
                endif()
            else()
                # Look for default ROCm installation
                file(GLOB _rocm_versions "C:/Program Files/AMD/ROCm/*")
                if(_rocm_versions)
                    list(SORT _rocm_versions ORDER DESCENDING)
                    list(GET _rocm_versions 0 _latest_rocm)
                    set(_hip_clang "${_latest_rocm}/lib/llvm/bin/clang++.exe")
                    if(NOT EXISTS "${_hip_clang}")
                        set(_hip_clang "${_latest_rocm}/bin/clang++.exe")
                    endif()
                endif()
            endif()
            
            if(EXISTS "${_hip_clang}")
                message(STATUS "HIP clang++ found: ${_hip_clang}")
                message(STATUS "")
                message(STATUS "=== IMPORTANT: Windows HIP Build Instructions ===")
                message(STATUS "For HIP builds on Windows, please re-run CMake with:")
                message(STATUS "  cmake -DCMAKE_CXX_COMPILER=\"${_hip_clang}\" ...")
                message(STATUS "Or use the build script: .\\build_lichtfeld.ps1 -GpuBackend HIP")
                message(STATUS "=================================================")
                message(STATUS "")
            else()
                message(WARNING "HIP SDK clang++ not found. Please install AMD HIP SDK.")
            endif()
        endif()
        
        project(LichtFeld-Studio LANGUAGES CXX C)
        message(STATUS "Building with AMD ROCm/HIP backend (Windows - using clang++)")
    else()
        # Linux: Use native HIP language support
        project(LichtFeld-Studio LANGUAGES HIP CXX C)
        message(STATUS "Building with AMD ROCm/HIP backend (Linux)")
    endif()
else()
    project(LichtFeld-Studio LANGUAGES CUDA CXX C)
    message(STATUS "Building with NVIDIA CUDA backend")
endif()

# Disable C++20 module scanning - clang-scan-deps causes file locking issues on Windows
# This must be set before any targets are created
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Add cmake module path for custom find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Get git commit hash (short and full)
execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH_SHORT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)

execute_process(
        COMMAND git describe --dirty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAGGED_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)

# Fallback if not in a git repo
if(NOT GIT_COMMIT_HASH_SHORT)
    set(GIT_COMMIT_HASH_SHORT "unknown")
endif()

if(NOT GIT_TAGGED_VERSION)
    set(GIT_TAGGED_VERSION "unknown")
endif()

message(STATUS "Git commit (short): ${GIT_COMMIT_HASH_SHORT}")
message(STATUS "Git commit (full): ${GIT_TAGGED_VERSION}")

# Setup
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

option(ENABLE_CUDA_GL_INTEROP "Enable CUDA-OpenGL interoperability" ON)
option(BUILD_TESTS "Build tests" OFF)

# Build fat binaries for all modern SMs (>= minimum). When OFF (default), build for the native GPU only.
option(BUILD_CUDA_ALL_SM "Build CUDA fat binaries targeting all modern SMs (>= minimum SM)" OFF)

# Minimum SM (compute capability) to include when BUILD_CUDA_ALL_SM=ON.
# Examples: 75, 80, 86, 89, 90, 100...
# Only has effect if BUILD_CUDA_ALL_SM is ON.
set(BUILD_CUDA_ALL_MIN_SM "86" CACHE STRING "Minimum SM (compute capability) used when BUILD_CUDA_ALL_SM=ON Examples: 75, 80, 86, 89...")

# HIP architecture selection for ROCm
# Common AMD GPU architectures:
# gfx906  - MI50/MI60 (Vega 20)
# gfx908  - MI100 (Arcturus)
# gfx90a  - MI200 series (Aldebaran)
# gfx940  - MI300X
# gfx942  - MI300A
# gfx1030 - RX 6800/6900 (RDNA 2)
# gfx1100 - RX 7900 (RDNA 3)
# gfx1101 - RX 7600/7700 (RDNA 3)
set(HIP_ARCHITECTURES "gfx90a;gfx940;gfx942;gfx1100;gfx1101" CACHE STRING "HIP GPU architectures to build for (ROCm)")

# Set Torch_DIR based on platform and build type
if (WIN32)
    if(USE_HIP)
        # Windows with HIP/ROCm - use ROCm LibTorch if available
        # First check if CMAKE_PREFIX_PATH contains PyTorch installation
        if(CMAKE_PREFIX_PATH)
            foreach(_prefix ${CMAKE_PREFIX_PATH})
                if(EXISTS "${_prefix}/share/cmake/Torch/TorchConfig.cmake")
                    set(Torch_DIR "${_prefix}/share/cmake/Torch")
                    message(STATUS "[${PROJECT_NAME}] Using PyTorch ROCm from CMAKE_PREFIX_PATH: ${Torch_DIR}")
                    break()
                endif()
            endforeach()
        endif()
        
        # If not found via CMAKE_PREFIX_PATH, try local directories
        if(NOT Torch_DIR)
            if(EXISTS "${PROJ_ROOT_DIR}/external/libtorch-rocm/share/cmake/Torch")
                set(Torch_DIR "${PROJ_ROOT_DIR}/external/libtorch-rocm/share/cmake/Torch")
                message(STATUS "[${PROJECT_NAME}] Using ROCm LibTorch from ${Torch_DIR}")
            elseif(EXISTS "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
                # Fallback to standard libtorch (may work with ROCm Python environment)
                set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
                message(STATUS "[${PROJECT_NAME}] Using standard LibTorch from ${Torch_DIR}")
                message(WARNING "Using CUDA LibTorch with HIP backend. This may not work correctly. Consider installing ROCm LibTorch.")
            else()
                message(FATAL_ERROR "LibTorch not found! For ROCm builds:\n"
                    "  1. Set CMAKE_PREFIX_PATH to your PyTorch ROCm installation, e.g.:\n"
                    "     -DCMAKE_PREFIX_PATH=\"C:/path/to/site-packages/torch\"\n"
                    "  2. Or place LibTorch ROCm in external/libtorch-rocm/")
            endif()
        endif()
    elseif(CMAKE_CONFIGURATION_TYPES AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
        # https://github.com/pytorch/pytorch/issues/155667
        set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
        message(STATUS "[${PROJECT_NAME}] User didn't specify -DCMAKE_BUILD_TYPE so use this path for searching libtorch -> ${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")

        if (NOT EXISTS "${PROJ_ROOT_DIR}/external/release/")
            message(FATAL_ERROR "you don't have a such path ${PROJ_ROOT_DIR}/external/release/ on your disk")
        endif()

        if (NOT EXISTS "${PROJ_ROOT_DIR}/external/release/libtorch/")
            message(FATAL_ERROR "you don't have an extracted pre-installed libtorch (${PROJ_ROOT_DIR}/external/release/libtorch)!")
        endif()
    else()
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
            message(STATUS "[${PROJECT_NAME}] User specified -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} so let's search libtorch by this path -> ${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")

            if (NOT EXISTS "${PROJ_ROOT_DIR}/external/release/")
                message(FATAL_ERROR "you don't have a such path ${PROJ_ROOT_DIR}/external/release/ on your disk")
            endif()

            if (NOT EXISTS "${PROJ_ROOT_DIR}/external/release/libtorch")
                message(FATAL_ERROR "you don't have an extracted pre-installed libtorch (${PROJ_ROOT_DIR}/external/release/libtorch)!")
            endif()
        elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(Torch_DIR "${PROJ_ROOT_DIR}/external/debug/libtorch/share/cmake/Torch")
            message(STATUS "[${PROJECT_NAME}] User specified -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} so let's search libtorch by this path -> ${PROJ_ROOT_DIR}/external/debug/libtorch/share/cmake/Torch")

            if (NOT EXISTS "${PROJ_ROOT_DIR}/external/debug/")
                message(FATAL_ERROR "you don't have a such path ${PROJ_ROOT_DIR}/external/debug/ on your disk")
            endif()

            if (NOT EXISTS "${PROJ_ROOT_DIR}/external/debug/libtorch")
                message(FATAL_ERROR "you don't have an extracted pre-installed libtorch (${PROJ_ROOT_DIR}/external/debug/libtorch)!")
            endif()
        else()
            message(FATAL_ERROR "libtorch binaries only available for Debug and Release on Windows. Current build type: '${CMAKE_BUILD_TYPE}'")
        endif()
    endif()
else()
    # Linux - check for ROCm PyTorch or CUDA PyTorch
    if(USE_HIP)
        # Try ROCm libtorch first
        if(EXISTS "${PROJ_ROOT_DIR}/external/libtorch-rocm/share/cmake/Torch")
            set(Torch_DIR "${PROJ_ROOT_DIR}/external/libtorch-rocm/share/cmake/Torch")
        elseif(EXISTS "${PROJ_ROOT_DIR}/external/libtorch/share/cmake/Torch")
            set(Torch_DIR "${PROJ_ROOT_DIR}/external/libtorch/share/cmake/Torch")
        endif()
    else()
        set(Torch_DIR "${PROJ_ROOT_DIR}/external/libtorch/share/cmake/Torch")
    endif()
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
elseif(USE_HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()

# Parallel build
include(ProcessorCount)
ProcessorCount(total_cores)
if(total_cores GREATER 1)
    math(EXPR used_cores "${total_cores} - 2")
    set(ENV{MAKEFLAGS} "-j${used_cores}")
    message(STATUS "Building with ${used_cores} cores")
endif()

# ============================================================================
# GPU Backend Setup
# ============================================================================

if(USE_CUDA)
# CUDA setup
# On Windows, nvidia-smi.exe is in System32, but CMake might be running as 32-bit
# which causes WOW64 redirection. Explicitly search in Sysnative to avoid this.
if(WIN32)
    find_program(NVIDIA_SMI nvidia-smi
        PATHS "C:/Windows/Sysnative" "C:/Windows/System32"
        NO_DEFAULT_PATH)
    if(NOT NVIDIA_SMI)
        find_program(NVIDIA_SMI nvidia-smi)
    endif()
else()
    find_program(NVIDIA_SMI nvidia-smi)
endif()
if(NVIDIA_SMI)
    execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE DETECTED_COMPUTE_CAP
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    message(STATUS "Detected GPU compute capability: ${DETECTED_COMPUTE_CAP}")
    set(LichtFeld-Studio_CUDA_ARCH "native")
else()
    message(WARNING "nvidia-smi not found! We are likely in CI, default to 8.6.")
    set(DETECTED_COMPUTE_CAP 8.6)
    set(LichtFeld-Studio_CUDA_ARCH "86")
endif()


# Detect supported SMs by querying 'ptxas --help' and filter by a minimum SM.
# We cannot use all or all-major since 5x and 6x versions are not supported
# in the CUDA code.
function(gs_detect_supported_sms OUT_VAR)
    set(options)
    set(oneValueArgs MIN_SM)
    cmake_parse_arguments(GS "${options}" "${oneValueArgs}" "" ${ARGN})

    # Use 'ptxas --help' to get the valid families
    set(_sms_list "")
    get_filename_component(_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    find_program(PTXAS_EXECUTABLE NAMES ptxas
            HINTS "${_cuda_bin}" "${_cuda_bin}/.." "${CUDAToolkit_BIN_DIR}"
            PATHS ENV PATH)

    if(PTXAS_EXECUTABLE)
        execute_process(COMMAND "${PTXAS_EXECUTABLE}" --help
                OUTPUT_VARIABLE _ptxas_help
                ERROR_VARIABLE  _ptxas_err
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_STRIP_TRAILING_WHITESPACE)
        set(_help "${_ptxas_help}\n${_ptxas_err}")

        string(REGEX MATCHALL "sm_[0-9]+" _matches "${_help}")
        if(_matches)
            string(REPLACE "sm_" "" _nums "${_matches}")
            string(REPLACE " "  ";" _nums "${_nums}")
            separate_arguments(_nums)
            list(REMOVE_DUPLICATES _nums)
            list(SORT _nums COMPARE NATURAL)
            foreach(n IN LISTS _nums)
                if(GS_MIN_SM)
                    if(n GREATER_EQUAL ${GS_MIN_SM})
                        list(APPEND _sms_list ${n})
                    endif()
                else()
                    list(APPEND _sms_list ${n})
                endif()
            endforeach()
        endif()
    endif()

    if(NOT _sms_list)
        set(_sms_list 75 80 86 87 89 90 100 101 120)
        if(GS_MIN_SM)
            set(_filtered "")
            foreach(n IN LISTS _sms_list)
                if(n GREATER_EQUAL ${GS_MIN_SM})
                    list(APPEND _filtered ${n})
                endif()
            endforeach()
            set(_sms_list "${_filtered}")
        endif()
    endif()

    set(${OUT_VAR} "${_sms_list}" PARENT_SCOPE)
endfunction()

if(DEFINED TORCH_CUDA_ARCH_LIST)
    unset(TORCH_CUDA_ARCH_LIST CACHE)
endif()

if(BUILD_CUDA_ALL_SM)
    # Validate BUILD_CUDA_ALL_MIN_SM
    if(NOT BUILD_CUDA_ALL_MIN_SM MATCHES "^[0-9]+$")
        message(FATAL_ERROR "BUILD_CUDA_ALL_MIN_SM must be an integer like 70, 75, 86, 90.")
    endif()

    # Enforce lower bound 70
    if(BUILD_CUDA_ALL_MIN_SM LESS 70)
        message(FATAL_ERROR "BUILD_CUDA_ALL_MIN_SM must be >= 70 when BUILD_CUDA_ALL_SM=ON.")
    endif()

    # Fat binary for all the families
    gs_detect_supported_sms(CUDA_SMS MIN_SM ${BUILD_CUDA_ALL_MIN_SM})
    if(CUDA_SMS)
        # Build Torch list with decimals and add +PTX only to the highest SM
        list(SORT CUDA_SMS COMPARE NATURAL)
        list(GET CUDA_SMS -1 _max_sm)
        set(_torch_list "")
        foreach(sm IN LISTS CUDA_SMS)
            math(EXPR _maj "${sm}/10")
            math(EXPR _min "${sm}%10")
            set(_ver "${_maj}.${_min}")
            if(sm EQUAL ${_max_sm})
                list(APPEND _torch_list "${_ver}+PTX")
            else()
                list(APPEND _torch_list "${_ver}")
            endif()
        endforeach()
        list(JOIN _torch_list ";" TORCH_CUDA_ARCH_LIST)
        set(TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}" CACHE STRING "Torch CUDA arch list" FORCE)

        # Per target CUDA architectures for CMake to be set after find_package(Torch)
        set(LichtFeld-Studio_CUDA_ARCH "${CUDA_SMS};${_max_sm}-virtual")
        message(STATUS "CUDA architectures (auto >=${BUILD_CUDA_ALL_MIN_SM}): ${LichtFeld-Studio_CUDA_ARCH}")
        message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
    else()
        message(FATAL_ERROR "Failed to detect any CUDA architecture >= ${BUILD_CUDA_ALL_MIN_SM}. Please check your CUDA toolkit installation.")
    endif()
else()
    # Native NVidia family only
    message(STATUS "CUDA architectures: ${LichtFeld-Studio_CUDA_ARCH} (native)")
    # Set Torch to the detected compute capability
    set(TORCH_CUDA_ARCH_LIST "${DETECTED_COMPUTE_CAP}" CACHE STRING "Torch CUDA arch list" FORCE)
    message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
endif()

enable_language(CUDA)
find_package(CUDAToolkit 12.8 REQUIRED)

# Create nvToolsExt target BEFORE finding Torch (PyTorch expects this)
if(NOT TARGET CUDA::nvToolsExt)
    find_library(NVTOOLSEXT_LIBRARY
            NAMES nvToolsExt
            PATHS
            ${CUDAToolkit_LIBRARY_ROOT}
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDAToolkit_TARGET_DIR}/lib64
            /usr/local/cuda-12.8/lib64
            PATH_SUFFIXES lib lib64
            NO_DEFAULT_PATH
    )

    if(NVTOOLSEXT_LIBRARY)
        add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
        set_target_properties(CUDA::nvToolsExt PROPERTIES
                IMPORTED_LOCATION ${NVTOOLSEXT_LIBRARY}
        )
        message(STATUS "Created CUDA::nvToolsExt target: ${NVTOOLSEXT_LIBRARY}")
    else()
        # Create empty interface library (nvToolsExt is deprecated in CUDA 12+)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        message(STATUS "nvToolsExt not found (normal for CUDA 12+), created empty interface target")
    endif()
endif()

message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} found at ${CUDAToolkit_TARGET_DIR}")

    # Set GPU library variables for later use
    set(GPU_RUNTIME_LIBRARY CUDA::cudart)
    set(GPU_RAND_LIBRARY CUDA::curand)
    set(GPU_BLAS_LIBRARY CUDA::cublas)
    set(GPU_SOLVER_LIBRARY CUDA::cusolver)
    set(GPU_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    set(GPU_ARCH_FLAG "${LichtFeld-Studio_CUDA_ARCH}")

elseif(USE_HIP)
    # ROCm/HIP setup
    if(WIN32)
        # Windows: HIP SDK uses clang++ instead of CMake's HIP language
        # Note: When using PyTorch ROCm, it provides hip::device and other targets
        # Find HIP SDK installation for include dirs and additional libraries
        if(DEFINED ENV{HIP_PATH})
            set(HIP_PATH "$ENV{HIP_PATH}")
        else()
            # Default HIP SDK installation path on Windows
            set(HIP_PATH "C:/Program Files/AMD/ROCm/6.2")
            if(NOT EXISTS "${HIP_PATH}")
                set(HIP_PATH "C:/Program Files/AMD/ROCm/6.1")
            endif()
            if(NOT EXISTS "${HIP_PATH}")
                # Try to find via registry or common paths
                file(GLOB HIP_PATHS "C:/Program Files/AMD/ROCm/*")
                if(HIP_PATHS)
                    list(GET HIP_PATHS -1 HIP_PATH)  # Get latest version
                endif()
            endif()
        endif()

        if(EXISTS "${HIP_PATH}")
            message(STATUS "HIP SDK found at: ${HIP_PATH}")
            
            # Set ROCM_PATH environment variable for PyTorch's LoadHIP.cmake
            # LoadHIP.cmake checks ROCM_PATH to find the ROCm installation
            set(ENV{ROCM_PATH} "${HIP_PATH}")
            
            # Convert HIP_PATH to CMake path format (forward slashes)
            file(TO_CMAKE_PATH "${HIP_PATH}" HIP_PATH_CMAKE)
            file(TO_CMAKE_PATH "${HIP_PATH}" HIP_PATH_CMAKE)
            
            # Get 8.3 short paths to avoid FindHIP.cmake issues with spaces in paths
            # This is necessary because FindHIP.cmake doesn't properly handle spaces in Windows paths
            string(REPLACE "/" "\\" HIP_PATH_NATIVE "${HIP_PATH}")
            execute_process(
                COMMAND cmd /c for %I in ("${HIP_PATH_NATIVE}") do @echo %~sI
                OUTPUT_VARIABLE HIP_PATH_SHORT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            file(TO_CMAKE_PATH "${HIP_PATH_SHORT}" HIP_PATH_CMAKE_SHORT)
            message(STATUS "HIP SDK short path: ${HIP_PATH_CMAKE_SHORT}")
            
            # Set HIP_CLANG_PATH before FindHIP.cmake to avoid path issues
            set(HIP_CLANG_PATH "${HIP_PATH_CMAKE_SHORT}/llvm/bin" CACHE PATH "HIP Clang Path" FORCE)
            set(ENV{HIP_CLANG_PATH} "${HIP_PATH_CMAKE_SHORT}/llvm/bin")
            
            # NOTE: Do NOT use find_package(HIP) on Windows!
            # AMD's FindHIP.cmake has path escaping issues that break compilation.
            # Instead, we manually set HIP variables and use our HIPUtils.cmake
            # which provides a working hip_add_library() macro.
            
            # Set HIP_FOUND manually (since we're not using find_package(HIP))
            set(HIP_FOUND TRUE)
            # HIP_VERSION needs to be in MAJOR.MINOR.PATCH format for AMD's scripts
            set(HIP_VERSION "7.1.0" CACHE STRING "HIP Version" FORCE)
            set(HIP_VERSION_MAJOR "7" CACHE STRING "" FORCE)
            set(HIP_VERSION_MINOR "1" CACHE STRING "" FORCE)
            set(HIP_VERSION_PATCH "0" CACHE STRING "" FORCE)
            
            message(STATUS "HIP SDK configured manually (avoiding AMD's FindHIP.cmake)")
            message(STATUS "HIP_CLANG_PATH: ${HIP_CLANG_PATH}")
            
            # Set up HIP compiler
            # HIP SDK 6.4+ places clang in lib/llvm/bin
            set(HIP_CLANG "${HIP_PATH}/lib/llvm/bin/clang++.exe")
            if(NOT EXISTS "${HIP_CLANG}")
                # Fallback to bin directory for older SDK versions
                set(HIP_CLANG "${HIP_PATH}/bin/clang++.exe")
            endif()
            if(NOT EXISTS "${HIP_CLANG}")
                set(HIP_CLANG "${HIP_PATH}/bin/hipcc.bat")
            endif()
            
            # Include directories - use short paths to avoid FindHIP.cmake issues
            set(HIP_INCLUDE_DIRS "${HIP_PATH_CMAKE_SHORT}/include")
            set(GPU_INCLUDE_DIRS "${HIP_INCLUDE_DIRS}")
            
            # Library directories - use short paths
            set(HIP_LIB_DIR "${HIP_PATH_CMAKE_SHORT}/lib")
            link_directories("${HIP_LIB_DIR}")
            
            # Find required libraries (for reference, PyTorch will provide targets)
            find_library(AMDHIP64_LIBRARY amdhip64 PATHS "${HIP_LIB_DIR}" NO_DEFAULT_PATH)
            find_library(HIPBLAS_LIBRARY hipblas PATHS "${HIP_LIB_DIR}" NO_DEFAULT_PATH)
            find_library(HIPRAND_LIBRARY hiprand PATHS "${HIP_LIB_DIR}" NO_DEFAULT_PATH)
            find_library(HIPSOLVER_LIBRARY hipsolver PATHS "${HIP_LIB_DIR}" NO_DEFAULT_PATH)
            
            if(NOT AMDHIP64_LIBRARY)
                message(FATAL_ERROR "amdhip64 library not found in ${HIP_LIB_DIR}")
            endif()
            
            # Do NOT create hip::device target here - PyTorch's find_package(HIP)
            # in LoadHIP.cmake will create it. Creating it here causes conflicts.
            # The target will be available after find_package(Torch)
            
            # Set GPU library variables (will be updated after Torch is loaded)
            set(GPU_RAND_LIBRARY "${HIPRAND_LIBRARY}")
            set(GPU_BLAS_LIBRARY "${HIPBLAS_LIBRARY}")
            if(HIPSOLVER_LIBRARY)
                set(GPU_SOLVER_LIBRARY "${HIPSOLVER_LIBRARY}")
            else()
                set(GPU_SOLVER_LIBRARY "")  # hipSOLVER may not be available
            endif()
            set(GPU_ARCH_FLAG "${HIP_ARCHITECTURES}")
            
            # Windows HIP compile flags
            set(HIP_CXX_FLAGS "-x hip --offload-arch=${HIP_ARCHITECTURES} -D__HIP_PLATFORM_AMD__")
            
            message(STATUS "HIP runtime: ${AMDHIP64_LIBRARY}")
            message(STATUS "HIP architectures: ${HIP_ARCHITECTURES}")
        else()
            message(FATAL_ERROR "HIP SDK not found! Please install AMD HIP SDK and set HIP_PATH environment variable.")
        endif()
    else()
        # Linux: Use native HIP language support
        enable_language(HIP)
        find_package(HIP REQUIRED)
        find_package(hipblas REQUIRED)
        find_package(hiprand REQUIRED)

        # Optionally find hipCUB/rocPRIM for CUB-style operations
        find_package(hipcub QUIET)
        find_package(rocprim QUIET)
        find_package(rocthrust QUIET)
        
        # Include HIPUtils.cmake for consistent hip_add_library() behavior
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HIPUtils.cmake)

        message(STATUS "ROCm/HIP ${HIP_VERSION} found")
        message(STATUS "HIP compiler: ${HIP_HIPCC_EXECUTABLE}")
        message(STATUS "HIP architectures: ${HIP_ARCHITECTURES}")

        # Set GPU library variables for later use
        set(GPU_RUNTIME_LIBRARY hip::device)
        if(TARGET roc::hiprand)
            set(GPU_RAND_LIBRARY roc::hiprand)
        else()
            set(GPU_RAND_LIBRARY hiprand)
        endif()
        if(TARGET roc::hipblas)
            set(GPU_BLAS_LIBRARY roc::hipblas)
        else()
            set(GPU_BLAS_LIBRARY hipblas)
        endif()
        # hipSOLVER for advanced linear algebra (optional)
        find_package(hipsolver QUIET)
        if(TARGET roc::hipsolver)
            set(GPU_SOLVER_LIBRARY roc::hipsolver)
        elseif(hipsolver_FOUND)
            set(GPU_SOLVER_LIBRARY hipsolver)
        else()
            set(GPU_SOLVER_LIBRARY "")  # hipSOLVER may not be available
        endif()
        set(GPU_INCLUDE_DIRS ${HIP_INCLUDE_DIRS})
        set(GPU_ARCH_FLAG "${HIP_ARCHITECTURES}")
    endif()

    # Add HIP compile definitions
    add_definitions(-D__HIP_PLATFORM_AMD__)
    add_definitions(-DUSE_ROCM)
    
    # On Windows with clang++, WIN32 preprocessor macro is not defined by default.
    # Standard MSVC defines WIN32 automatically, but clang++ only defines _WIN32.
    # Many Windows-specific code paths in this project use #ifdef WIN32.
    if(WIN32)
        add_definitions(-DWIN32)
    endif()
endif()

# For HIP builds on Windows, we need to call find_package(hip CONFIG) BEFORE find_package(Torch)
# This is because PyTorch's LoadHIP.cmake also calls find_package(hip CONFIG), and AMD ROCm's
# hip-targets.cmake will fail if some targets (hip::host, hip::device) are already partially defined.
# By calling it first here, all HIP targets are properly created before Torch loads.
if(USE_HIP AND WIN32)
    # Set HIP_PLATFORM before calling find_package(hip CONFIG)
    # This is required by AMD's hip-config.cmake
    set(HIP_PLATFORM "amd" CACHE STRING "HIP platform (amd or nvidia)")
    set(ENV{HIP_PLATFORM} "amd")
    
    # Ensure HIP path is in CMAKE_PREFIX_PATH for CONFIG mode
    if(DEFINED ENV{HIP_PATH})
        file(TO_CMAKE_PATH "$ENV{HIP_PATH}" _hip_path)
        list(APPEND CMAKE_PREFIX_PATH "${_hip_path}")
    endif()
    
    # Call find_package(hip CONFIG) to create all HIP targets properly
    # This must happen before find_package(Torch) to avoid target conflicts
    find_package(hip CONFIG QUIET)
    if(hip_FOUND)
        message(STATUS "AMD ROCm hip CONFIG found (pre-Torch): ${hip_VERSION}")
    endif()
    
    # Include our Windows-compatible HIPUtils.cmake AFTER find_package(hip CONFIG)
    # This overrides AMD's HIP_ADD_LIBRARY macro with our working version
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HIPUtils.cmake)
endif()

# Find packages
find_package(Torch REQUIRED)

# For Windows HIP builds with clang, we need to filter out MSVC-style flags from Torch
# that are incompatible with clang (e.g., /permissive-, /EHsc, /bigobj, etc.)
if(USE_HIP AND WIN32)
    # Use Python script as compiler launcher to filter MSVC flags
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(MSVC_FLAG_FILTER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/filter_msvc_flags.py")
    
    # Set compiler launcher to filter MSVC flags from PyTorch
    set(CMAKE_CXX_COMPILER_LAUNCHER "${Python3_EXECUTABLE}" "${MSVC_FLAG_FILTER_SCRIPT}")
    set(CMAKE_C_COMPILER_LAUNCHER "${Python3_EXECUTABLE}" "${MSVC_FLAG_FILTER_SCRIPT}")
    
    message(STATUS "Windows HIP build: Using MSVC flag filter: ${MSVC_FLAG_FILTER_SCRIPT}")
    
    # For Windows HIP builds, set the ROCm path for clang to find device libraries
    # Use 8.3 short path format to avoid issues with spaces
    set(HIP_ROCM_PATH "C:/PROGRA~1/AMD/ROCm/7.10")
    set(HIP_DEVICE_LIB_PATH "${HIP_ROCM_PATH}/lib/llvm/amdgcn/bitcode")
    
    # Set CMAKE_CXX_FLAGS to include ROCm paths
    # IMPORTANT: Add -fno-openmp to prevent clang's OpenMP wrappers from conflicting
    # with MSVC STL type_traits (causes vectorcall redefinition errors)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --rocm-path=${HIP_ROCM_PATH} --rocm-device-lib-path=${HIP_DEVICE_LIB_PATH} -fno-openmp")
    
    # Add global compile options for HIP builds
    # Force include hip_runtime_compat.h to intercept cuda_runtime.h includes
    # and provide CUDA->HIP API mappings for all translation units
    # Note: hip_runtime_compat.h includes hip_warp_override.h to handle warp sync functions
    add_compile_options(
        "-Wno-unused-command-line-argument"
        "-include" "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels/hip_runtime_compat.h"
    )
    
    # Add cuda_shim directory as highest priority include path
    # This intercepts #include <cuda_runtime.h> etc. and redirects to HIP equivalents
    include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_shim")
    
    # Create a helper function to apply HIP settings to a target
    function(lfs_filter_torch_options_for_cxx TARGET_NAME)
        # This function is now mostly a placeholder since we use compiler launcher
        # But can be used for additional per-target customization if needed
    endfunction()
endif()

find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glad REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(args REQUIRED)

# Find fmt BEFORE spdlog to ensure we use vcpkg's fmt version
# This is critical because PyTorch bundles fmt v12, but vcpkg's spdlog.lib is built with fmt v11
# By finding fmt first and adding its include path with high priority, we ensure consistency
find_package(fmt REQUIRED)

# For HIP builds, ensure vcpkg's fmt include path comes before PyTorch's bundled fmt
# PyTorch includes fmt v12 headers, but vcpkg's spdlog.lib is linked against fmt v11
# This causes linker errors due to ABI incompatibility (fmt::v11 vs fmt::v12)
if(USE_HIP AND WIN32)
    get_target_property(_fmt_inc fmt::fmt INTERFACE_INCLUDE_DIRECTORIES)
    if(_fmt_inc)
        include_directories(BEFORE SYSTEM ${_fmt_inc})
        message(STATUS "Added vcpkg fmt include path with high priority: ${_fmt_inc}")
    endif()
endif()

find_package(spdlog REQUIRED)
find_package(Freetype REQUIRED)
find_package(WebP REQUIRED)
find_package(LibArchive REQUIRED)
find_package(OpenImageIO REQUIRED)

# Set GPU_RUNTIME_LIBRARY after find_package(Torch) for HIP builds
# PyTorch's LoadHIP.cmake creates the hip::device target
if(USE_HIP AND WIN32)
    if(TARGET hip::device)
        set(GPU_RUNTIME_LIBRARY hip::device)
        message(STATUS "Using hip::device target from PyTorch")
    else()
        message(WARNING "hip::device target not found after loading Torch. HIP linking may fail.")
    endif()
endif()

# Set architectures based on backend
if(USE_CUDA)
    # TORCH_CUDA_ARCH_LIST is already defined by Torch
    # use it here as the global default CUDA architecture list too
    if(LichtFeld-Studio_CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}" CACHE STRING "CUDA architectures" FORCE)
    endif()
elseif(USE_HIP)
    set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCHITECTURES}" CACHE STRING "HIP architectures" FORCE)
endif()

if(WIN32)
    if(USE_CUDA)
        set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared") # when using INTEROP in windows static and dynamic symbols are mixed
    endif()
endif()

# Check for GPU-OpenGL interop capability (only if enabled)
if(ENABLE_CUDA_GL_INTEROP)
    include(CheckCXXSourceCompiles)
    if(USE_CUDA)
        set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})
        set(CMAKE_REQUIRED_LIBRARIES ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES})
        check_cxx_source_compiles("
            #ifdef _WIN32
            #include <windows.h>
            #endif
            #include <cuda_runtime.h>
            #include <cuda_gl_interop.h>
            int main() {
                cudaGraphicsResource_t resource;
                return 0;
            }
        " CUDA_GL_INTEROP_FOUND)
    elseif(USE_HIP)
        set(CMAKE_REQUIRED_INCLUDES ${HIP_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})
        set(CMAKE_REQUIRED_LIBRARIES ${HIP_LIBRARIES} ${OPENGL_LIBRARIES})
        check_cxx_source_compiles("
            #include <hip/hip_runtime.h>
            #include <hip/hip_gl_interop.h>
            int main() {
                hipGraphicsResource_t resource;
                return 0;
            }
        " CUDA_GL_INTEROP_FOUND)
    endif()

    if(CUDA_GL_INTEROP_FOUND)
        message(STATUS "GPU-OpenGL interop support: ENABLED")
        set(CUDA_GL_INTEROP_ENABLED 1)
    else()
        message(WARNING "GPU-OpenGL interop support: DISABLED (headers not found)")
        set(CUDA_GL_INTEROP_ENABLED 0)
    endif()
else()
    message(STATUS "GPU-OpenGL interop support: DISABLED (by user option)")
    set(CUDA_GL_INTEROP_FOUND FALSE)
    set(CUDA_GL_INTEROP_ENABLED 0)
endif()

# Create a configuration header
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/config.h"
        @ONLY
)

# Project-wide definitions
add_definitions(-DPROJECT_ROOT_PATH="${PROJ_ROOT_DIR}")

if(USE_CUDA)
    add_definitions(-DUSE_CUDA=1)
    add_definitions(-DUSE_HIP=0)
elseif(USE_HIP)
    add_definitions(-DUSE_CUDA=0)
    add_definitions(-DUSE_HIP=1)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    if(USE_CUDA)
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
    elseif(USE_HIP)
        set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} -g -O0")
    endif()
endif()

# =============================================================================
# Kernel libs
# =============================================================================
add_subdirectory(gsplat)
add_subdirectory(src/training/fastgs)

if(USE_CUDA)
    set_target_properties(gsplat_backend PROPERTIES CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}")
    set_target_properties(fastgs_backend PROPERTIES CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}")
elseif(USE_HIP)
    set_target_properties(gsplat_backend PROPERTIES HIP_ARCHITECTURES "${HIP_ARCHITECTURES}")
    set_target_properties(fastgs_backend PROPERTIES HIP_ARCHITECTURES "${HIP_ARCHITECTURES}")
endif()

# =============================================================================
# SHARED CUDA KERNELS (only if needed)
# =============================================================================
set(KERNEL_SOURCES)
if(CUDA_GL_INTEROP_FOUND)
    list(APPEND KERNEL_SOURCES kernels/cuda_gl_interop.cu)
endif()

# Add SOG-related kernels
list(APPEND KERNEL_SOURCES
        kernels/morton_encoding.cu
        kernels/kmeans.cu
)

# Only create gaussian_kernels if there are kernels
if(KERNEL_SOURCES)
    if(USE_CUDA)
        add_library(gs_kernels STATIC ${KERNEL_SOURCES})
        set_target_properties(gs_kernels PROPERTIES
                CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
                CUDA_SEPARABLE_COMPILATION ON
                POSITION_INDEPENDENT_CODE ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    elseif(USE_HIP)
        # For HIP, use lfs_hip_add_library from HIPUtils.cmake for Windows-compatible HIP support
        set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -std=c++20")
        lfs_hip_add_library(gs_kernels STATIC ${KERNEL_SOURCES})
        set_target_properties(gs_kernels PROPERTIES
                POSITION_INDEPENDENT_CODE ON
        )
    endif()

    target_include_directories(gs_kernels
            PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels
            ${GPU_INCLUDE_DIRS}
            ${OPENGL_INCLUDE_DIRS}
            PRIVATE
    )

    target_link_libraries(gs_kernels
            PUBLIC
            ${GPU_RUNTIME_LIBRARY}
            ${GPU_RAND_LIBRARY}
            ${GPU_BLAS_LIBRARY}
            ${TORCH_LIBRARIES}
            glm::glm
            ${OPENGL_LIBRARIES}
            spdlog::spdlog
    )

    if(USE_CUDA)
        target_compile_options(gs_kernels PRIVATE
                # CUDA device code + MSVC host compiler flags (Windows only)
                $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo -Xcompiler=/Od -Xcompiler=/Z7>
                $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 -use_fast_math --ptxas-options=-v -Xcompiler=/O2 -Xcompiler=/DNDEBUG>

                # CUDA device code for non-Windows
                $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo>
                $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -use_fast_math --ptxas-options=-v>
        )
    elseif(USE_HIP)
        target_compile_options(gs_kernels PRIVATE
                $<$<COMPILE_LANGUAGE:HIP>:$<$<CONFIG:Debug>:-O0 -g>>
                $<$<COMPILE_LANGUAGE:HIP>:$<$<CONFIG:Release>:-O3 -ffast-math>>
        )
    endif()
else()
    # Create empty interface library if no kernels
    add_library(gs_kernels INTERFACE)
endif()

# =============================================================================
# LIBRARY MODULES
# =============================================================================
add_subdirectory(src/core_new)     # New core module (lfs::core namespace)
add_subdirectory(src/training_new) # New training module (lfs::training namespace)
add_subdirectory(src/loader_new)   # New loader module (lfs::loader namespace)
add_subdirectory(src/core)         # Foundation module
add_subdirectory(src/geometry)     # Geometry utilities
add_subdirectory(src/project)      # Project management
add_subdirectory(src/loader)       # Data loading
add_subdirectory(src/training)     # Training functionality
add_subdirectory(src/rendering)   # Rendering pipeline
add_subdirectory(src/visualizer)  # GUI and visualization

# =============================================================================
# MAIN EXECUTABLE
# =============================================================================
add_executable(${PROJECT_NAME} src/main.cpp resources/lichtfeld-studio.rc)

if(USE_CUDA)
    set_target_properties(${PROJECT_NAME} PROPERTIES
            CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
elseif(USE_HIP)
    set_target_properties(${PROJECT_NAME} PROPERTIES
            HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
    )
endif()

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include

        ${GPU_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        gs_core
        gs_geometry
        gs_project
        gs_loader
        gs_training
        gs_rendering
        gs_visualizer
        gs_kernels
        gsplat_backend
        fastgs_backend

        ${OPENGL_LIBRARIES}
        ${GPU_RUNTIME_LIBRARY}
        spdlog::spdlog
        taywee::args
        OpenImageIO::OpenImageIO
)

# Platform-specific settings
if(WIN32)
    file(GLOB TORCH_DLLS "${Torch_DIR}/../../../lib/*.dll")
    foreach(TORCH_DLL ${TORCH_DLLS})
        add_custom_command(
                TARGET ${PROJECT_NAME}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_DLL}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endforeach()
elseif(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE GL GLU)

    find_path(TORCH_LIB_DIR libtorch_cpu.so
            PATHS "${Torch_DIR}/../../../lib"
            NO_DEFAULT_PATH)

    if(TORCH_LIB_DIR)
        if(USE_CUDA)
            set_target_properties(${PROJECT_NAME} PROPERTIES
                    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${TORCH_LIB_DIR}"
                    BUILD_WITH_INSTALL_RPATH TRUE
                    INSTALL_RPATH_USE_LINK_PATH TRUE
            )
        elseif(USE_HIP)
            set_target_properties(${PROJECT_NAME} PROPERTIES
                    INSTALL_RPATH "$ENV{ROCM_PATH}/lib:${TORCH_LIB_DIR}"
                    BUILD_WITH_INSTALL_RPATH TRUE
                    INSTALL_RPATH_USE_LINK_PATH TRUE
            )
        endif()
        message(STATUS "Torch library directory: ${TORCH_LIB_DIR}")
    else()
        message(WARNING "Could not find Torch library directory")
    endif()
endif()

# Build type configuration function
function(configure_build_type target)
    get_target_property(target_type ${target} TYPE)

    if(target_type STREQUAL "INTERFACE_LIBRARY")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${target} INTERFACE DEBUG_BUILD)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${target} INTERFACE RELEASE_BUILD)
        endif()
    else()
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${target} PRIVATE DEBUG_BUILD)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${target} PRIVATE RELEASE_BUILD)
        endif()
    endif()
endfunction()

# Apply build configuration to all targets
configure_build_type(gs_core)
configure_build_type(gs_kernels)
configure_build_type(gsplat_backend)
configure_build_type(fastgs_backend)
configure_build_type(${PROJECT_NAME})

# =============================================================================
# TESTING (Optional)
# =============================================================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Tests enabled. Build with 'ninja lichtfeld_tests' and run with 'ninja run_tests' or 'ctest'")
endif()

# =============================================================================
# BUILD INFO & OPTIMIZATIONS
# =============================================================================
message(STATUS "===========================================")
message(STATUS "Build Configuration:")
if(USE_CUDA)
    message(STATUS "  GPU Backend: CUDA")
    message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA Archs: ${CMAKE_CUDA_ARCHITECTURES}")
elseif(USE_HIP)
    message(STATUS "  GPU Backend: ROCm/HIP")
    message(STATUS "  HIP Version: ${HIP_VERSION}")
    message(STATUS "  HIP Archs: ${HIP_ARCHITECTURES}")
endif()
message(STATUS "  Torch Version: ${Torch_VERSION}")
message(STATUS "  OpenGL Found: ${OPENGL_FOUND}")
message(STATUS "  FreeType Found: ${FREETYPE_FOUND}")
message(STATUS "  WebP Found: ${WebP_FOUND}")
message(STATUS "  LibArchive Found: ${LibArchive_FOUND}")
message(STATUS "  GPU-GL Interop Option: ${ENABLE_CUDA_GL_INTEROP}")
message(STATUS "  GPU-GL Interop Available: ${CUDA_GL_INTEROP_FOUND}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
if(USE_CUDA)
    message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
elseif(USE_HIP)
    message(STATUS "  HIP Standard: ${CMAKE_HIP_STANDARD}")
endif()
message(STATUS "  OpenImageIO Found: ${OpenImageIO_FOUND}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "===========================================")

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()