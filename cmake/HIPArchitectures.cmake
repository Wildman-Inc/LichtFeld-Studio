# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

function(lfs_configure_amdgpu_architectures output_variable)
    if(CMAKE_HOST_WIN32)
        set(_lfs_default_amdgpu_arch "gfx1151")
    else()
        set(_lfs_default_amdgpu_arch "native")
    endif()

    set(LFS_AMDGPU_ARCH "${_lfs_default_amdgpu_arch}" CACHE STRING
        "AMDGPU target architectures (semicolon- or comma-separated)")
    string(REPLACE "," ";" _lfs_amdgpu_archs "${LFS_AMDGPU_ARCH}")
    list(TRANSFORM _lfs_amdgpu_archs STRIP)
    list(FILTER _lfs_amdgpu_archs EXCLUDE REGEX "^$")
    list(REMOVE_DUPLICATES _lfs_amdgpu_archs)
    if(NOT _lfs_amdgpu_archs)
        message(FATAL_ERROR
            "LFS_AMDGPU_ARCH must contain at least one AMDGPU architecture")
    endif()

    if(NOT CMAKE_HOST_WIN32)
        set(CMAKE_HIP_ARCHITECTURES "${_lfs_amdgpu_archs}" CACHE STRING
            "HIP target architectures" FORCE)
    endif()

    set(${output_variable} "${_lfs_amdgpu_archs}" PARENT_SCOPE)
endfunction()
