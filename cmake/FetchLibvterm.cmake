# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Libvterm.cmake
# libvterm - a VT220/xterm terminal emulator library (MIT license)
# Located in external/libvterm as git submodule

set(LIBVTERM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/libvterm)
set(LIBVTERM_ENC_DIR ${LIBVTERM_SOURCE_DIR}/src/encoding)

# If the submodule is not present, fetch libvterm directly.
if(NOT EXISTS "${LIBVTERM_SOURCE_DIR}/include/vterm.h" OR NOT EXISTS "${LIBVTERM_SOURCE_DIR}/src/vterm.c")
    message(STATUS "libvterm submodule missing; fetching https://github.com/neovim/libvterm")
    include(FetchContent)
    FetchContent_Declare(
        lfs_libvterm
        GIT_REPOSITORY https://github.com/neovim/libvterm.git
        GIT_TAG v0.3.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(lfs_libvterm)
    if(NOT DEFINED lfs_libvterm_SOURCE_DIR OR "${lfs_libvterm_SOURCE_DIR}" STREQUAL "")
        message(FATAL_ERROR "Failed to fetch libvterm source directory via FetchContent")
    endif()
    set(LIBVTERM_SOURCE_DIR ${lfs_libvterm_SOURCE_DIR})
    set(LIBVTERM_ENC_DIR ${LIBVTERM_SOURCE_DIR}/src/encoding)
endif()

# Generate encoding tables if not present (pre-generated in submodule)
if(NOT EXISTS ${LIBVTERM_ENC_DIR}/DECdrawing.inc OR NOT EXISTS ${LIBVTERM_ENC_DIR}/uk.inc)
    find_package(Perl REQUIRED)
endif()

if(NOT EXISTS ${LIBVTERM_ENC_DIR}/DECdrawing.inc)
    execute_process(
        COMMAND ${PERL_EXECUTABLE} -CSD ${LIBVTERM_SOURCE_DIR}/tbl2inc_c.pl ${LIBVTERM_ENC_DIR}/DECdrawing.tbl
        OUTPUT_FILE ${LIBVTERM_ENC_DIR}/DECdrawing.inc
        WORKING_DIRECTORY ${LIBVTERM_SOURCE_DIR}
        RESULT_VARIABLE _decdrawing_gen_result
        ERROR_VARIABLE _decdrawing_gen_error
    )
    if(NOT _decdrawing_gen_result EQUAL 0)
        message(FATAL_ERROR "Failed to generate DECdrawing.inc via tbl2inc_c.pl: ${_decdrawing_gen_error}")
    endif()
endif()

if(NOT EXISTS ${LIBVTERM_ENC_DIR}/uk.inc)
    execute_process(
        COMMAND ${PERL_EXECUTABLE} -CSD ${LIBVTERM_SOURCE_DIR}/tbl2inc_c.pl ${LIBVTERM_ENC_DIR}/uk.tbl
        OUTPUT_FILE ${LIBVTERM_ENC_DIR}/uk.inc
        WORKING_DIRECTORY ${LIBVTERM_SOURCE_DIR}
        RESULT_VARIABLE _uk_gen_result
        ERROR_VARIABLE _uk_gen_error
    )
    if(NOT _uk_gen_result EQUAL 0)
        message(FATAL_ERROR "Failed to generate uk.inc via tbl2inc_c.pl: ${_uk_gen_error}")
    endif()
endif()

set(LIBVTERM_SOURCES
    ${LIBVTERM_SOURCE_DIR}/src/encoding.c
    ${LIBVTERM_SOURCE_DIR}/src/keyboard.c
    ${LIBVTERM_SOURCE_DIR}/src/mouse.c
    ${LIBVTERM_SOURCE_DIR}/src/parser.c
    ${LIBVTERM_SOURCE_DIR}/src/pen.c
    ${LIBVTERM_SOURCE_DIR}/src/screen.c
    ${LIBVTERM_SOURCE_DIR}/src/state.c
    ${LIBVTERM_SOURCE_DIR}/src/unicode.c
    ${LIBVTERM_SOURCE_DIR}/src/vterm.c
)

add_library(vterm STATIC ${LIBVTERM_SOURCES})

target_include_directories(vterm
    PUBLIC ${LIBVTERM_SOURCE_DIR}/include
    PRIVATE ${LIBVTERM_SOURCE_DIR}/src
)

set_target_properties(vterm PROPERTIES
    C_STANDARD 99
    POSITION_INDEPENDENT_CODE ON
)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vterm PRIVATE -w)
elseif(MSVC)
    target_compile_options(vterm PRIVATE /w)
endif()
