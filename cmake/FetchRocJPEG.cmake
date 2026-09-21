# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

set(_lfs_rocjpeg_default OFF)
if(WIN32 AND USE_HIP)
    set(_lfs_rocjpeg_default ON)
endif()
option(LFS_ENABLE_ROCJPEG "Decode JPEG images with the native Windows D3D11/VCN rocJPEG backend" ${_lfs_rocjpeg_default})
if(NOT LFS_ENABLE_ROCJPEG)
    return()
endif()
if(NOT WIN32 OR NOT USE_HIP)
    message(FATAL_ERROR "LFS_ENABLE_ROCJPEG currently requires Windows and LFS_GPU_BACKEND=HIP")
endif()

include(FetchContent)
set(ROCM_PATH "${LFS_ROCM_PATH}")
set(ROCJPEG_ROCM_DEVEL_PATH "${LFS_ROCM_DEVEL_PATH}")
set(GPU_TARGETS "${HIP_ARCHITECTURES}")
FetchContent_Declare(rocjpeg
    GIT_REPOSITORY https://github.com/Yasei-no-otoko/rocJPEG.git
    GIT_TAG 4b1d92d448d421406647978d6c9722f28a8a08ac
    GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(rocjpeg)
