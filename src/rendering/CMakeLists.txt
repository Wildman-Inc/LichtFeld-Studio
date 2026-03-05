# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_subdirectory(rasterizer)

set(RENDERING_SOURCES
        rendering_engine.cpp
        rendering_engine_impl.cpp
        rendering_pipeline.cpp
        gs_rasterizer_tensor.cpp
        point_cloud_renderer.cpp
        grid_renderer.cpp
        bbox_renderer.cpp
        ellipsoid_renderer.cpp
        axes_renderer.cpp
        pivot_renderer.cpp
        text_renderer.cpp
        viewport_gizmo.cpp
        framebuffer_factory.cpp
        shader_manager.cpp
        screen_renderer.cpp
        split_view_renderer.cpp
        camera_frustum_renderer.cpp
        mesh_renderer.cpp
        mesh2splat.cpp
        depth_compositor.cpp
        # Needed even when interop is disabled:
        # - CudaGLInteropTextureImpl<false> definitions (cuda_gl_interop.cpp)
        cuda_gl_interop.cpp
)

if(USE_HIP AND WIN32)
    # ROCm Windows currently fails compiling cuda_gl_kernels.cu due HIP/cmath
    # header conflicts. Use a no-op fallback so rendering/visualizer can link.
    list(APPEND RENDERING_SOURCES cuda_gl_kernels_stub.cpp)
else()
    list(APPEND RENDERING_SOURCES cuda_gl_kernels.cu)
endif()

# When interop is enabled, cuda_gl_kernels.cu also exports GL interop kernels.
if(CUDA_GL_INTEROP_ENABLED)
    message(STATUS "Rendering: GPU-GL interop kernels enabled")
else()
    message(STATUS "Rendering: GPU-GL interop kernels disabled (fallback mode)")
endif()

if(USE_HIP)
    lfs_hip_add_library(lfs_rendering STATIC ${RENDERING_SOURCES})
else()
    add_library(lfs_rendering STATIC ${RENDERING_SOURCES})
endif()

set_target_properties(lfs_rendering PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(lfs_rendering
        PUBLIC
        lfs_core                # Link core module
        lfs_geometry            # LibTorch-free geometry module
        # gs_kernels              # Temporarily disabled - requires LibTorch
        PRIVATE
        lfs_rasterizer
        glm::glm
        glad::glad
        ${OPENGL_LIBRARIES}
        ${GPU_RUNTIME_LIBRARY}
        spdlog::spdlog
        Freetype::Freetype
)

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
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
)

if(NOT BUILD_PORTABLE)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rendering/rendering.hpp
            DESTINATION include/rendering)
endif()
