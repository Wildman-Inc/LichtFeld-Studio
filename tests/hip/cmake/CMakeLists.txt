# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.24)

set(LFS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH
    "LichtFeld Studio source directory")
get_filename_component(LFS_SOURCE_DIR "${LFS_SOURCE_DIR}" ABSOLUTE)

include("${LFS_SOURCE_DIR}/cmake/HIPArchitectures.cmake")
lfs_configure_amdgpu_architectures(_lfs_cdna_architectures)

project(LichtFeldCDNACompileSmoke LANGUAGES CXX HIP)

set(HIP_ARCHITECTURES "${_lfs_cdna_architectures}" CACHE STRING
    "HIP target architectures" FORCE)
include("${LFS_SOURCE_DIR}/cmake/HIPUtils.cmake")

set(_lfs_cdna_smoke_source
    "${LFS_SOURCE_DIR}/tests/hip/cdna_wave_smoke.hip.cpp")
set(_lfs_cdna_include_dirs
    "${LFS_SOURCE_DIR}/include/cuda_shim"
    "${LFS_SOURCE_DIR}/src/core/include")

lfs_hip_add_library(lfs_cdna_core_wave OBJECT "${_lfs_cdna_smoke_source}")
target_include_directories(lfs_cdna_core_wave BEFORE PRIVATE
    ${_lfs_cdna_include_dirs}
    "${LFS_SOURCE_DIR}/src/core/tensor")

lfs_hip_add_library(lfs_cdna_training_wave OBJECT "${_lfs_cdna_smoke_source}")
target_compile_definitions(lfs_cdna_training_wave PRIVATE
    LFS_TEST_TRAINING_WARP_REDUCE=1)
target_include_directories(lfs_cdna_training_wave BEFORE PRIVATE
    ${_lfs_cdna_include_dirs}
    "${LFS_SOURCE_DIR}/src/training/include")

lfs_hip_add_library(lfs_cdna_vksplat_input OBJECT
    "${LFS_SOURCE_DIR}/src/visualizer/rendering/vksplat_input_packer_cuda.cu")
target_include_directories(lfs_cdna_vksplat_input BEFORE PRIVATE
    ${_lfs_cdna_include_dirs}
    "${LFS_SOURCE_DIR}/src/visualizer/rendering")

lfs_hip_add_library(lfs_cdna_lod_dequant OBJECT
    "${LFS_SOURCE_DIR}/src/visualizer/rendering/lod_page_dequant_cuda.cu")
set_property(SOURCE
    "${LFS_SOURCE_DIR}/src/visualizer/rendering/lod_page_dequant_cuda.cu"
    APPEND PROPERTY COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
target_include_directories(lfs_cdna_lod_dequant BEFORE PRIVATE
    ${_lfs_cdna_include_dirs}
    "${LFS_SOURCE_DIR}/src"
    "${LFS_SOURCE_DIR}/src/visualizer/include"
    "${LFS_SOURCE_DIR}/src/visualizer/rendering")

lfs_hip_add_library(lfs_cdna_rad_quant OBJECT
    "${LFS_SOURCE_DIR}/src/io/cuda/rad_encode_quant.cu")
target_include_directories(lfs_cdna_rad_quant BEFORE PRIVATE
    ${_lfs_cdna_include_dirs}
    "${LFS_SOURCE_DIR}/src"
    "${LFS_SOURCE_DIR}/src/io")

foreach(_target IN ITEMS
        lfs_cdna_core_wave
        lfs_cdna_training_wave
        lfs_cdna_vksplat_input
        lfs_cdna_lod_dequant
        lfs_cdna_rad_quant)
    get_target_property(_target_architectures "${_target}" HIP_ARCHITECTURES)
    if(NOT "${_target_architectures}" STREQUAL "${_lfs_cdna_architectures}")
        message(FATAL_ERROR
            "${_target}: HIP_ARCHITECTURES='${_target_architectures}', expected '${_lfs_cdna_architectures}'")
    endif()
endforeach()

message(STATUS "CDNA compile smoke architectures: ${_lfs_cdna_architectures}")
