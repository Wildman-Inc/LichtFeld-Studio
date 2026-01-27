# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.30)

set(FASTLFS_SOURCES
        # rasterizer
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/rasterization_api.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/forward.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/backward.cu
        # optimizer
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/src/adam_api.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/src/adam.cu
)

# Create library based on GPU backend (upstream target name)
if(USE_HIP)
        # Use lfs_hip_add_library from HIPUtils.cmake for Windows-compatible HIP support
        set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -std=c++20")
        lfs_hip_add_library(fastlfs_backend STATIC ${FASTLFS_SOURCES})
else()
        add_library(fastlfs_backend STATIC ${FASTLFS_SOURCES})
endif()

# PTX-only builds require separable compilation OFF
if(LFS_DEFER_DEVICE_SYMBOLS)
        set(_sep_comp OFF)
else()
        set(_sep_comp ON)
endif()

if(USE_CUDA)
        set_target_properties(fastlfs_backend PROPERTIES
                        CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
                        CUDA_SEPARABLE_COMPILATION ${_sep_comp}
                        POSITION_INDEPENDENT_CODE ON
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
elseif(USE_HIP)
        set_target_properties(fastlfs_backend PROPERTIES
                        POSITION_INDEPENDENT_CODE ON
        )
endif()

target_include_directories(fastlfs_backend
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/include
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CMAKE_SOURCE_DIR}/include
        ${GPU_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        PRIVATE
)

# Pure CUDA libraries only
target_link_libraries(fastlfs_backend
        PUBLIC
        ${GPU_RUNTIME_LIBRARY}   # CUDA::cudart or hip::device
        ${GPU_RAND_LIBRARY}      # CUDA::curand or hipRAND
        ${GPU_BLAS_LIBRARY}      # CUDA::cublas or hipBLAS
        ${TORCH_LIBRARIES}
)

if(TARGET lfs_core_cuda)
    target_link_libraries(fastlfs_backend PUBLIC lfs_core_cuda)
endif()

target_compile_options(fastlfs_backend PRIVATE
  # CUDA device code + MSVC host compiler flags (Windows only)
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo -Xcompiler=/Od -Xcompiler=/Z7>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221 -Xcompiler=/O2 -Xcompiler=/DNDEBUG>

  # CUDA on Windows: Pass /utf-8 to MSVC when invoked by nvcc
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:
    -Xcompiler=/utf-8
    -Xcompiler=/D_CRT_SECURE_NO_WARNINGS
    --diag-suppress=27  # Suppress "character value out of range" warnings from fmt
  >

  # HIP compile options
  $<$<COMPILE_LANGUAGE:HIP>:$<$<CONFIG:Debug>:-O0 -g>>
  $<$<COMPILE_LANGUAGE:HIP>:$<$<CONFIG:Release>:-O3 -ffast-math>>

  # C++ MSVC
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Z7>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /DNDEBUG>

  # C++ non-MSVC (GCC/Clang/AppleClang/IntelLLVM)
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

# Windows: Disable fmt Unicode check for CUDA files (nvcc doesn't properly propagate /utf-8)
if(MSVC)
    target_compile_definitions(fastlfs_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(fastlfs_backend PRIVATE _DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(fastlfs_backend PRIVATE RELEASE_BUILD)
endif()
