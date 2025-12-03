# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.30)

# TODO: This file is a bit of a mess and needs to be cleaned up.

# Get torch from parent
if(NOT DEFINED TORCH_INCLUDE_DIRS)
    find_package(Torch REQUIRED)
endif()

# All fastgs sources together
set(FASTGS_SOURCES
        # rasterizer
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/rasterization_api.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/forward.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/src/backward.cu
        # optimizer
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/src/adam_api.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/src/adam.cu
)

# Create library based on GPU backend
if(USE_HIP)
    # Use lfs_hip_add_library from HIPUtils.cmake for Windows-compatible HIP support
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -std=c++20")
    lfs_hip_add_library(fastgs_backend STATIC ${FASTGS_SOURCES})
else()
    add_library(fastgs_backend STATIC ${FASTGS_SOURCES})
endif()

if(USE_CUDA)
    set_target_properties(fastgs_backend PROPERTIES
            CUDA_ARCHITECTURES "${gaussian_splatting_cuda_CUDA_ARCH}"
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
elseif(USE_HIP)
    set_target_properties(fastgs_backend PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
endif()

target_include_directories(fastgs_backend
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/include
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CMAKE_SOURCE_DIR}/include
        ${GPU_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        PRIVATE
)

target_link_libraries(fastgs_backend
        PUBLIC
        ${GPU_RUNTIME_LIBRARY}   # CUDA::cudart or hip::device
        ${GPU_RAND_LIBRARY}      # CUDA::curand or hipRAND
        ${GPU_BLAS_LIBRARY}      # CUDA::cublas or hipBLAS
        ${TORCH_LIBRARIES}
)

target_compile_options(fastgs_backend PRIVATE
  # CUDA device code + MSVC host compiler flags (Windows only)
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo -Xcompiler=/Od -Xcompiler=/Z7>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221 -Xcompiler=/O2 -Xcompiler=/DNDEBUG>

  # CUDA device code for non-Windows
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221>

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

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(fastgs_backend PRIVATE _DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(fastgs_backend PRIVATE RELEASE_BUILD)
endif()
