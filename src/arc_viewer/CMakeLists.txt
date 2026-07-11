# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

find_package(SDL3 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glm CONFIG REQUIRED)

if(COMMAND lfs_validate_linux_sdl_video_backends)
    lfs_validate_linux_sdl_video_backends()
endif()

set(_LFS_ARC_GLSLC "${Vulkan_GLSLC_EXECUTABLE}")
if(NOT _LFS_ARC_GLSLC OR NOT EXISTS "${_LFS_ARC_GLSLC}")
    find_program(_LFS_ARC_GLSLC NAMES glslc
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
endif()

if(NOT _LFS_ARC_GLSLC)
    set(_LFS_ARC_GLSLANG "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
    if(NOT _LFS_ARC_GLSLANG OR NOT EXISTS "${_LFS_ARC_GLSLANG}")
        find_program(_LFS_ARC_GLSLANG NAMES glslangValidator
            HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    endif()
endif()

if(NOT _LFS_ARC_GLSLC AND NOT _LFS_ARC_GLSLANG)
    message(FATAL_ERROR
        "Arc Viewer shaders require glslc or glslangValidator at build time. "
        "Install the Vulkan SDK or place either compiler on PATH.")
endif()

set(_LFS_ARC_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}")
set(_LFS_ARC_VERTEX_SHADER "${_LFS_ARC_SHADER_OUTPUT_DIR}/point.vert.spv")
set(_LFS_ARC_FRAGMENT_SHADER "${_LFS_ARC_SHADER_OUTPUT_DIR}/point.frag.spv")

function(lfs_arc_compile_shader source_file output_file)
    if(_LFS_ARC_GLSLC)
        add_custom_command(
            OUTPUT "${output_file}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_LFS_ARC_SHADER_OUTPUT_DIR}"
            COMMAND "${_LFS_ARC_GLSLC}" -O --target-env=vulkan1.0
                -o "${output_file}" "${source_file}"
            DEPENDS "${source_file}"
            COMMENT "Compiling Arc Viewer shader ${source_file}"
            VERBATIM)
    else()
        add_custom_command(
            OUTPUT "${output_file}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_LFS_ARC_SHADER_OUTPUT_DIR}"
            COMMAND "${_LFS_ARC_GLSLANG}" -V --target-env vulkan1.0
                -o "${output_file}" "${source_file}"
            DEPENDS "${source_file}"
            COMMENT "Compiling Arc Viewer shader ${source_file}"
            VERBATIM)
    endif()
endfunction()

lfs_arc_compile_shader(
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/point.vert"
    "${_LFS_ARC_VERTEX_SHADER}")
lfs_arc_compile_shader(
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/point.frag"
    "${_LFS_ARC_FRAGMENT_SHADER}")

add_custom_target(lfs_arc_viewer_shaders
    DEPENDS "${_LFS_ARC_VERTEX_SHADER}" "${_LFS_ARC_FRAGMENT_SHADER}")

add_executable(LichtFeld-Studio-Arc-Viewer
    main.cpp
    arc_viewer.cpp
    ply_loader.cpp)

add_dependencies(LichtFeld-Studio-Arc-Viewer lfs_arc_viewer_shaders)

target_compile_features(LichtFeld-Studio-Arc-Viewer PRIVATE cxx_std_20)
set_target_properties(LichtFeld-Studio-Arc-Viewer PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(_LFS_ARC_CONFIGURATION DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_property(TARGET LichtFeld-Studio-Arc-Viewer PROPERTY
        "RUNTIME_OUTPUT_DIRECTORY_${_LFS_ARC_CONFIGURATION}" "${CMAKE_BINARY_DIR}")
endforeach()

target_compile_definitions(LichtFeld-Studio-Arc-Viewer PRIVATE
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    $<$<PLATFORM_ID:Windows>:WIN32_LEAN_AND_MEAN>)

if(MSVC)
    target_compile_options(LichtFeld-Studio-Arc-Viewer PRIVATE
        /W4 /permissive- /utf-8)
else()
    target_compile_options(LichtFeld-Studio-Arc-Viewer PRIVATE
        -Wall -Wextra -Wpedantic)
endif()

target_link_libraries(LichtFeld-Studio-Arc-Viewer PRIVATE
    SDL3::SDL3
    Vulkan::Vulkan
    glm::glm)

if(WIN32)
    set(_LFS_ARC_SDL_TARGET SDL3::SDL3)
    get_target_property(_LFS_ARC_SDL_ALIAS SDL3::SDL3 ALIASED_TARGET)
    if(_LFS_ARC_SDL_ALIAS)
        set(_LFS_ARC_SDL_TARGET "${_LFS_ARC_SDL_ALIAS}")
    endif()
    get_target_property(_LFS_ARC_SDL_TYPE "${_LFS_ARC_SDL_TARGET}" TYPE)

    get_filename_component(_LFS_ARC_VULKAN_LIBRARY_DIR "${Vulkan_LIBRARY}" DIRECTORY)
    get_filename_component(_LFS_ARC_VULKAN_PREFIX "${_LFS_ARC_VULKAN_LIBRARY_DIR}" DIRECTORY)
    find_file(_LFS_ARC_VULKAN_LOADER_DLL NAMES vulkan-1.dll
        HINTS "${_LFS_ARC_VULKAN_PREFIX}/bin" "$ENV{VULKAN_SDK}/Bin"
        NO_DEFAULT_PATH)
    if(_LFS_ARC_VULKAN_LOADER_DLL)
        add_custom_command(TARGET LichtFeld-Studio-Arc-Viewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_LFS_ARC_VULKAN_LOADER_DLL}"
                "$<TARGET_FILE_DIR:LichtFeld-Studio-Arc-Viewer>"
            COMMENT "Staging the Vulkan loader beside the Arc Viewer executable")
    else()
        message(WARNING
            "No redistributable vulkan-1.dll was found; the Arc Viewer package "
            "will require the system Vulkan loader supplied by the GPU driver.")
    endif()
endif()

if(WIN32 AND _LFS_ARC_SDL_TYPE STREQUAL "SHARED_LIBRARY")
    add_custom_command(TARGET LichtFeld-Studio-Arc-Viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SDL3::SDL3>"
            "$<TARGET_FILE_DIR:LichtFeld-Studio-Arc-Viewer>"
        COMMENT "Staging SDL3 beside the Arc Viewer executable")
endif()

install(TARGETS LichtFeld-Studio-Arc-Viewer
    RUNTIME DESTINATION .
    COMPONENT ArcViewer)
install(FILES
    "${_LFS_ARC_VERTEX_SHADER}"
    "${_LFS_ARC_FRAGMENT_SHADER}"
    DESTINATION .
    COMPONENT ArcViewer)

if(WIN32 AND _LFS_ARC_SDL_TYPE STREQUAL "SHARED_LIBRARY")
    install(FILES "$<TARGET_FILE:SDL3::SDL3>"
        DESTINATION .
        COMPONENT ArcViewer)
endif()
if(WIN32 AND _LFS_ARC_VULKAN_LOADER_DLL)
    install(FILES "${_LFS_ARC_VULKAN_LOADER_DLL}"
        DESTINATION .
        COMPONENT ArcViewer)
endif()
if(MSVC)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ".")
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_COMPONENT ArcViewer)
    include(InstallRequiredSystemLibraries)
    foreach(_LFS_ARC_MSVC_RUNTIME IN LISTS CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
        add_custom_command(TARGET LichtFeld-Studio-Arc-Viewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_LFS_ARC_MSVC_RUNTIME}"
                "$<TARGET_FILE_DIR:LichtFeld-Studio-Arc-Viewer>"
            COMMENT "Staging MSVC runtime for the Arc Viewer")
    endforeach()
endif()

install(FILES "${PROJECT_SOURCE_DIR}/LICENSE"
    DESTINATION .
    RENAME LICENSE.txt
    COMPONENT ArcViewer)
install(FILES "${PROJECT_SOURCE_DIR}/THIRD_PARTY_LICENSES.md"
    DESTINATION .
    COMPONENT ArcViewer)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    DESTINATION .
    COMPONENT ArcViewer)
set(_LFS_ARC_NOTICE_DEPENDENCIES sdl3 glm vulkan-loader vulkan-headers)
foreach(_LFS_ARC_DEPENDENCY IN LISTS _LFS_ARC_NOTICE_DEPENDENCIES)
    set(_LFS_ARC_NOTICE_FILE
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/${_LFS_ARC_DEPENDENCY}/copyright")
    if(BUILD_PORTABLE AND NOT EXISTS "${_LFS_ARC_NOTICE_FILE}")
        message(FATAL_ERROR
            "Portable Arc Viewer packages require the ${_LFS_ARC_DEPENDENCY} notice: "
            "${_LFS_ARC_NOTICE_FILE}")
    endif()
    if(EXISTS "${_LFS_ARC_NOTICE_FILE}")
        install(FILES
            "${_LFS_ARC_NOTICE_FILE}"
            DESTINATION licenses
            RENAME "${_LFS_ARC_DEPENDENCY}.txt"
            COMPONENT ArcViewer)
    endif()
endforeach()

if(BUILD_TESTING)
    add_executable(lfs_arc_ply_loader_tests
        tests/ply_loader_tests.cpp
        ply_loader.cpp)
    target_compile_features(lfs_arc_ply_loader_tests PRIVATE cxx_std_20)
    target_include_directories(lfs_arc_ply_loader_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    if(MSVC)
        target_compile_options(lfs_arc_ply_loader_tests PRIVATE
            /W4 /permissive- /utf-8)
    else()
        target_compile_options(lfs_arc_ply_loader_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME arc_viewer.ply_loader COMMAND lfs_arc_ply_loader_tests)
endif()

message(STATUS "Arc Viewer product: SDL3 + Vulkan + glm")
message(STATUS "Arc Viewer shader compiler: ${_LFS_ARC_GLSLC}${_LFS_ARC_GLSLANG}")
