# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_subdirectory(rasterizer)

set(LFS_HAS_VKSPLAT OFF)
set(LFS_VKSPLAT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/vksplat/vksplat")
if(LFS_ENABLE_VKSPLAT AND EXISTS "${LFS_VKSPLAT_SOURCE_DIR}/src/gs_renderer.h")
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        add_library(lfs_vksplat STATIC
                ${LFS_VKSPLAT_SOURCE_DIR}/src/buffer.cpp
                ${LFS_VKSPLAT_SOURCE_DIR}/src/gs_pipeline.cpp
                ${LFS_VKSPLAT_SOURCE_DIR}/src/gs_renderer.cpp
                ${LFS_VKSPLAT_SOURCE_DIR}/src/perf_timer.cpp
                ${LFS_VKSPLAT_SOURCE_DIR}/src/scheduler.cpp
        )
        target_include_directories(lfs_vksplat PUBLIC "${LFS_VKSPLAT_SOURCE_DIR}/src")
        target_link_libraries(lfs_vksplat PUBLIC Vulkan::Vulkan glm::glm)
        target_compile_features(lfs_vksplat PUBLIC cxx_std_17)
        target_compile_definitions(lfs_vksplat PRIVATE
                $<$<PLATFORM_ID:Windows>:NOMINMAX>
                $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
        )
        set(LFS_HAS_VKSPLAT ON)
        message(STATUS "vksplat Vulkan rasterization enabled: ${LFS_VKSPLAT_SOURCE_DIR}")
    else()
        message(WARNING "LFS_ENABLE_VKSPLAT is ON but Vulkan SDK was not found; vksplat disabled")
    endif()
endif()

set(RENDERING_SOURCES
        rendering_engine.cpp
        rendering_engine_impl.cpp
        rendering_pipeline.cpp
        gs_rasterizer_tensor.cpp
        vksplat_rasterizer.cpp
        point_cloud_renderer.cpp
        grid_renderer.cpp
        bbox_renderer.cpp
        ellipsoid_renderer.cpp
        axes_renderer.cpp
        pivot_renderer.cpp
        text_renderer.cpp
        environment_renderer.cpp
        viewport_gizmo.cpp
        framebuffer_factory.cpp
        shader_manager.cpp
        screen_renderer.cpp
        split_view_renderer.cpp
        cuda_gl_interop.cpp
        camera_frustum_renderer.cpp
        mesh_renderer.cpp
        mesh2splat.cpp
        depth_compositor.cpp
)

# CUDA GL interop kernels are optional, but the saturation adjustment kernel is
# also implemented here and is used by the visualizer whenever a GPU backend is enabled.
if(USE_CUDA OR USE_HIP)
    list(APPEND RENDERING_SOURCES cuda_gl_kernels.cu)
endif()

lfs_add_gpu_library(lfs_rendering STATIC ${RENDERING_SOURCES})

target_link_libraries(lfs_rendering
        PUBLIC
        lfs_core                # Link core module
        lfs_geometry            # LibTorch-free geometry module
        # gs_kernels              # Temporarily disabled - requires LibTorch
        PRIVATE
        lfs_io
        lfs_rasterizer
        glm::glm
        glad::glad
        ${OPENGL_LIBRARIES}
        ${GPU_RUNTIME_LIBRARY}
        OpenImageIO::OpenImageIO
        spdlog::spdlog
        Freetype::Freetype
)

if(LFS_HAS_VKSPLAT)
    target_link_libraries(lfs_rendering PRIVATE lfs_vksplat)
endif()

# Since we're linking to lfs_core PUBLIC, we inherit its include directories
# which include: ${CMAKE_BINARY_DIR}/include (for config.h)
#                ${CMAKE_SOURCE_DIR}/gsplat (for Common.h)
target_include_directories(lfs_rendering
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${OPENGL_INCLUDE_DIRS}
)

# Handle shaders
set(RENDERING_BUILD_RESOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/resources")
set(RENDERING_SOURCE_RESOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")

file(MAKE_DIRECTORY "${RENDERING_BUILD_RESOURCE_DIR}/shaders")
file(MAKE_DIRECTORY "${RENDERING_BUILD_RESOURCE_DIR}/shaders/mesh2splat")

file(GLOB SHADER_FILES "${RENDERING_SOURCE_RESOURCE_DIR}/shaders/*")
foreach(SHADER_FILE ${SHADER_FILES})
    if(NOT IS_DIRECTORY ${SHADER_FILE})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
        configure_file(${SHADER_FILE} "${RENDERING_BUILD_RESOURCE_DIR}/shaders/${SHADER_NAME}" COPYONLY)
    endif()
endforeach()

file(GLOB MESH2SPLAT_SHADER_FILES "${RENDERING_SOURCE_RESOURCE_DIR}/shaders/mesh2splat/*")
foreach(SHADER_FILE ${MESH2SPLAT_SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
    configure_file(${SHADER_FILE} "${RENDERING_BUILD_RESOURCE_DIR}/shaders/mesh2splat/${SHADER_NAME}" COPYONLY)
endforeach()

target_compile_definitions(lfs_rendering PRIVATE
        SHADER_PATH="${RENDERING_BUILD_RESOURCE_DIR}/shaders"
        RENDERING_SOURCE_SHADER_PATH="${RENDERING_SOURCE_RESOURCE_DIR}/shaders"
        $<$<BOOL:${CUDA_GL_INTEROP_ENABLED}>:CUDA_GL_INTEROP_ENABLED>
        $<$<BOOL:${LFS_HAS_VKSPLAT}>:LFS_HAS_VKSPLAT=1>
        $<$<BOOL:${LFS_HAS_VKSPLAT}>:LFS_VKSPLAT_SHADER_PATH="${LFS_VKSPLAT_SOURCE_DIR}/shader/">
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
)

if(NOT BUILD_PORTABLE)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/rendering.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/frame_contract.hpp
            DESTINATION include/rendering)
endif()
