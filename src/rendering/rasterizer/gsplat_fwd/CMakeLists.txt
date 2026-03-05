cmake_minimum_required(VERSION 3.30)

# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Forward-only gsplat for viewer (no backward pass, no training dependencies)

# Ensure CUDA targets are available from parent
if(NOT TARGET CUDA::cudart)
    message(FATAL_ERROR "CUDA::cudart not found - ensure parent CMakeLists.txt sets up CUDA properly")
endif()

# All forward-only gsplat sources
set(GSPLAT_FWD_SOURCES
        # C++ files
        Intersect.cpp
        Projection.cpp
        Rasterization.cpp
        SphericalHarmonics.cpp

        # CUDA files
        IntersectTile.cu
        ProjectionUT3DGSFused.cu
        RasterizeToPixelsFromWorld3DGSFwd.cu
        SphericalHarmonicsCUDA.cu
)

# Forward-only gsplat backend for rendering
if(USE_HIP)
    lfs_hip_add_library(gsplat_fwd STATIC ${GSPLAT_FWD_SOURCES})
else()
    add_library(gsplat_fwd STATIC ${GSPLAT_FWD_SOURCES})
endif()

set_target_properties(gsplat_fwd PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
)
if(USE_CUDA)
    set_target_properties(gsplat_fwd PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
            CUDA_SEPARABLE_COMPILATION OFF
    )
endif()

target_include_directories(gsplat_fwd
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${GPU_INCLUDE_DIRS}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
)

target_link_libraries(gsplat_fwd
        PUBLIC
        ${GPU_RUNTIME_LIBRARY}
        glm::glm
)

target_compile_options(gsplat_fwd PRIVATE
  # CUDA device code + MSVC host compiler flags (Windows only)
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g ${LFS_CUDA_DEVICE_DEBUG_FLAGS} -lineinfo --expt-relaxed-constexpr -Xcompiler=/Od -Xcompiler=/Z7>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221 -Xcompiler=/O2 -Xcompiler=/DNDEBUG>

  # CUDA device code for non-Windows
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g ${LFS_CUDA_DEVICE_DEBUG_FLAGS} -lineinfo --expt-relaxed-constexpr>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221>

  # C++ MSVC
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Z7>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /DNDEBUG>

  # C++ no-MSVC (GCC/Clang/AppleClang/IntelLLVM)
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gsplat_fwd PRIVATE DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(gsplat_fwd PRIVATE RELEASE_BUILD)
endif()

# GLM workarounds for CUDA 12.x compatibility
target_compile_definitions(gsplat_fwd PRIVATE
    GLM_FORCE_CUDA
    GLM_COMPILER=0
)

message(STATUS "gsplat_fwd configuration:")
message(STATUS "  - Sources: ${GSPLAT_FWD_SOURCES}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
