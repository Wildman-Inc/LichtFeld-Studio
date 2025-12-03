cmake_minimum_required(VERSION 3.30)

# Get torch from parent
if(NOT DEFINED TORCH_INCLUDE_DIRS)
    find_package(Torch REQUIRED)
endif()

# Ensure GPU runtime targets are available from parent
if(USE_CUDA)
    if(NOT TARGET CUDA::cudart)
        message(FATAL_ERROR "CUDA::cudart not found - ensure parent CMakeLists.txt sets up CUDA properly")
    endif()

    if(NOT TARGET CUDA::curand)
        message(FATAL_ERROR "CUDA::curand not found - ensure parent CMakeLists.txt sets up CUDA properly")
    endif()

    if(NOT TARGET CUDA::cublas)
        message(FATAL_ERROR "CUDA::cublas not found - ensure parent CMakeLists.txt sets up CUDA properly")
    endif()
elseif(USE_HIP)
    if(NOT TARGET hip::device)
        message(FATAL_ERROR "hip::device not found - ensure parent CMakeLists.txt sets up HIP properly")
    endif()
endif()

# All gsplat sources together
set(GSPLAT_SOURCES
        # C++ files
        Intersect.cpp
        Projection.cpp
        QuatToRotmat.cpp
        Rasterization.cpp
        Relocation.cpp
        SphericalHarmonics.cpp

        # CUDA files
        IntersectTile.cu
        ProjectionUT3DGSFused.cu
        QuatToRotmatCUDA.cu
        RasterizeToPixelsFromWorld3DGSBwd.cu
        RasterizeToPixelsFromWorld3DGSFwd.cu
        RelocationCUDA.cu
        SphericalHarmonicsCUDA.cu
)

# Create library based on GPU backend
if(USE_HIP)
    # Use lfs_hip_add_library from HIPUtils.cmake for Windows-compatible HIP support
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -std=c++20")
    lfs_hip_add_library(gsplat_backend STATIC ${GSPLAT_SOURCES})
    set_target_properties(gsplat_backend PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(gsplat_backend STATIC ${GSPLAT_SOURCES})
endif()

if(USE_CUDA)
    set_target_properties(gsplat_backend PROPERTIES
            CXX_STANDARD 23
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${gaussian_splatting_cuda_CUDA_ARCH}"
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

target_include_directories(gsplat_backend
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
        ${GPU_INCLUDE_DIRS}
        PRIVATE
)

# Link libraries
# For HIP builds on Windows, we use LINK_ONLY to prevent Torch's MSVC compile flags
# from propagating to our clang-compiled HIP code
if(USE_HIP AND WIN32)
    target_link_libraries(gsplat_backend
            PUBLIC
            $<LINK_ONLY:${TORCH_LIBRARIES}>
            ${GPU_RUNTIME_LIBRARY}
            ${GPU_RAND_LIBRARY}
            ${GPU_BLAS_LIBRARY}
            glm::glm
    )
else()
    target_link_libraries(gsplat_backend
            PUBLIC
            ${TORCH_LIBRARIES}
            ${GPU_RUNTIME_LIBRARY}
            ${GPU_RAND_LIBRARY}
            ${GPU_BLAS_LIBRARY}
            glm::glm
    )
endif()

if(USE_CUDA)
    target_compile_options(gsplat_backend PRIVATE
      # CUDA device code + MSVC host compiler flags (Windows only)
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo -Xcompiler=/Od -Xcompiler=/Z7>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221 -Xcompiler=/O2 -Xcompiler=/DNDEBUG>

      # CUDA device code for non-Windows
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221>

      # C++ MSVC
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Z7>
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /DNDEBUG>

      # C++ no-MSVC (GCC/Clang/AppleClang/IntelLLVM)
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g>
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -DNDEBUG>
    )
endif()
# HIP compile options are handled by hip_add_library through HIP_HIPCC_FLAGS

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gsplat_backend PRIVATE DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(gsplat_backend PRIVATE RELEASE_BUILD)
endif()

# Print debug info
message(STATUS "gsplat_backend configuration:")
if(USE_CUDA)
    message(STATUS "  - GPU backend: CUDA")
    message(STATUS "  - CUDA targets available: ${gaussian_splatting_cuda_CUDA_ARCH}")
elseif(USE_HIP)
    message(STATUS "  - GPU backend: ROCm/HIP")
    message(STATUS "  - HIP targets available: ${HIP_ARCHITECTURES}")
endif()
message(STATUS "  - Sources: ${GSPLAT_SOURCES}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - C++ Standard: 23")
if(USE_CUDA)
    message(STATUS "  - CUDA Standard: 20")
    message(STATUS "  - CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
elseif(USE_HIP)
    message(STATUS "  - HIP Standard: 20")
    message(STATUS "  - HIP include dirs: ${HIP_INCLUDE_DIRS}")
endif()
message(STATUS "  - Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "  - Torch libraries: ${TORCH_LIBRARIES}")
