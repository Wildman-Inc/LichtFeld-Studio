# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# ============================================================================
# lfs_core_cuda - Unified CUDA library for core module
# Contains: Memory Arena, Lanczos Resize, Core Kernels (kmeans, morton)
# This library is PRIVATE to the core module
# ============================================================================

set(LFS_CORE_CUDA_SOURCES
    memory_arena.cu
    lanczos_resize/lanczos_resize.cu
    kernels/kmeans_new.cu
    kernels/kdtree_kmeans.cu
    kernels/morton_encoding_new.cu
    tensor_debug.cu
)

if(USE_HIP)
    # Use lfs_hip_add_library from HIPUtils.cmake for Windows-compatible HIP support
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -std=c++20")
    lfs_hip_add_library(lfs_core_cuda STATIC ${LFS_CORE_CUDA_SOURCES})
else()
    add_library(lfs_core_cuda STATIC ${LFS_CORE_CUDA_SOURCES})
endif()

target_include_directories(lfs_core_cuda
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src  # For core/cuda/* path
        ${GPU_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}  # For local includes within cuda/
        ${CMAKE_SOURCE_DIR}/src/core/tensor/internal  # For memory_pool.hpp and other internal headers
)

find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(lfs_core_cuda
    PUBLIC
        lfs_tensor      # Tensor library needed by lanczos_resize and kernels
        ${GPU_RUNTIME_LIBRARY}
        OpenMP::OpenMP_CXX
)

if(USE_CUDA)
    target_link_libraries(lfs_core_cuda PUBLIC CUDA::cuda_driver)
endif()

if(USE_CUDA)
    set_target_properties(lfs_core_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
elseif(USE_HIP)
    set_target_properties(lfs_core_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# CUDA compiler options
target_compile_options(lfs_core_cuda PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo --extended-lambda --expt-relaxed-constexpr -Xcompiler=/Od -Xcompiler=/Z7 -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS --diag-suppress=27>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr --ptxas-options=-v -Xcompiler=/O2 -Xcompiler=/DNDEBUG -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS --diag-suppress=27>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:GNU>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr --ptxas-options=-v>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-O0 -g -G -lineinfo --extended-lambda --expt-relaxed-constexpr>

    # HIP compile options
    $<$<COMPILE_LANGUAGE:HIP>:$<$<CONFIG:Debug>:-O0 -g>>
    $<$<COMPILE_LANGUAGE:HIP>:$<$<CONFIG:Release>:-O3 -ffast-math>>
)

# Windows: Disable fmt Unicode check for CUDA files
if(MSVC)
    target_compile_definitions(lfs_core_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
endif()

message(STATUS "✓ lfs_core_cuda library configured (memory arena, lanczos resize, core kernels)")
