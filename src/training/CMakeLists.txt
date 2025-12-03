# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Collect all training sources
set(TRAINING_HOST_SOURCES
        trainer.cpp
        training_setup.cpp

        # Rasterization
        rasterization/rasterizer.cpp
        rasterization/fast_rasterizer.cpp
        rasterization/rasterizer_autograd.cpp
        rasterization/fast_rasterizer_autograd.cpp

        # Strategies
        strategies/strategy_utils.cpp
        strategies/default_strategy.cpp
        strategies/mcmc.cpp

        # Optimizers
        optimizers/scheduler.cpp
        optimizers/fused_adam.cpp

        # Metrics
        metrics/metrics.cpp

        # Components
        components/bilateral_grid.cpp
        components/poseopt.cpp
        components/sparsity_optimizer.cpp
)

# CUDA kernel sources
set(TRAINING_KERNEL_SOURCES
        kernels/bilateral_grid_forward.cu
        kernels/bilateral_grid_backward.cu
        kernels/bilateral_grid_tv.cu
        kernels/ssim.cu
        kernels/regularization.cu
)

# Create training kernels library based on GPU backend
if(USE_HIP)
    # Use lfs_hip_add_library from HIPUtils.cmake for Windows-compatible HIP support
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -std=c++20")
    lfs_hip_add_library(gs_training_kernels STATIC ${TRAINING_KERNEL_SOURCES})
else()
    add_library(gs_training_kernels STATIC ${TRAINING_KERNEL_SOURCES})
endif()

if(USE_CUDA)
    set_target_properties(gs_training_kernels PROPERTIES
            CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
elseif(USE_HIP)
    set_target_properties(gs_training_kernels PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
endif()

target_include_directories(gs_training_kernels
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/gsplat
        ${CMAKE_CURRENT_SOURCE_DIR}/fastgs
        ${GPU_INCLUDE_DIRS}
)

target_link_libraries(gs_training_kernels
        PUBLIC
        ${GPU_RUNTIME_LIBRARY}   # CUDA::cudart or hip::device
        ${GPU_RAND_LIBRARY}      # CUDA::curand or hipRAND
        ${GPU_BLAS_LIBRARY}      # CUDA::cublas or hipBLAS
        ${TORCH_LIBRARIES}
        glm::glm
        spdlog::spdlog
)

target_compile_options(gs_training_kernels PRIVATE
        # CUDA device code + MSVC host compiler flags (Windows only)
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo -Xcompiler=/Od -Xcompiler=/Z7>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 -use_fast_math --ptxas-options=-v -Xcompiler=/O2 -Xcompiler=/DNDEBUG>

        # CUDA device code for non-Windows
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -use_fast_math --ptxas-options=-v>

        # HIP compile options
        $<$<COMPILE_LANGUAGE:HIP>:$<$<CONFIG:Debug>:-O0 -g>>
        $<$<COMPILE_LANGUAGE:HIP>:$<$<CONFIG:Release>:-O3 -ffast-math>>
)

# Create training host library
add_library(gs_training STATIC ${TRAINING_HOST_SOURCES})

target_include_directories(gs_training
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/gsplat
        ${CMAKE_CURRENT_SOURCE_DIR}/fastgs
        ${GPU_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
)

target_link_libraries(gs_training
        PUBLIC
        gs_core
        gs_training_kernels
        gs_kernels
        gs_loader
        gsplat_backend
        fastgs_backend
        ${TORCH_LIBRARIES}
        TBB::tbb
        nlohmann_json::nlohmann_json
        glm::glm
        Threads::Threads
        ${GPU_RUNTIME_LIBRARY}   # CUDA::cudart or hip::device
        spdlog::spdlog
        PRIVATE
        ${OPENGL_LIBRARIES}
)

# Compiler options
if (MSVC)
    target_compile_options(gs_training PRIVATE
            $<$<CONFIG:Debug>:/Od /Z7>
            $<$<CONFIG:Release>:/O2 /DNDEBUG>)
else ()
    target_compile_options(gs_training PRIVATE
            $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer -DDEBUG>
            $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>)
endif ()

# AVX2 support
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if (COMPILER_SUPPORTS_AVX2)
        target_compile_options(gs_training PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )
        target_compile_definitions(gs_training PRIVATE HAS_AVX2_SUPPORT)
    endif ()
endif ()

# Set properties
set_target_properties(gs_training PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        DEBUG_POSTFIX d
)
