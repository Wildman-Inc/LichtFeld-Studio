# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Windows ROCm HIP SDK may ship amd_comgr libraries/headers without a CMake package.
# Provide a lightweight finder so consumers (e.g. PyTorch LoadHIP.cmake) can proceed.

include(FindPackageHandleStandardArgs)

set(_amd_comgr_rocm_hints "")
if(DEFINED ROCM_PATH AND NOT "${ROCM_PATH}" STREQUAL "")
    list(APPEND _amd_comgr_rocm_hints "${ROCM_PATH}")
endif()
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    list(APPEND _amd_comgr_rocm_hints "$ENV{ROCM_PATH}")
endif()
if(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
    list(APPEND _amd_comgr_rocm_hints "$ENV{HIP_PATH}")
endif()
list(APPEND _amd_comgr_rocm_hints "C:/Program Files/AMD/ROCm/7.1" "C:/Program Files/AMD/ROCm/7.10")

find_path(amd_comgr_INCLUDE_DIR
    NAMES amd_comgr/amd_comgr.h
    HINTS ${_amd_comgr_rocm_hints}
    PATH_SUFFIXES include
)

find_library(amd_comgr_LIBRARY
    NAMES amd_comgr amd_comgr_3 amd_comgr0701
    HINTS ${_amd_comgr_rocm_hints}
    PATH_SUFFIXES lib lib64
)

if(amd_comgr_LIBRARY AND NOT TARGET amd_comgr::amd_comgr)
    add_library(amd_comgr::amd_comgr UNKNOWN IMPORTED)
    set_target_properties(amd_comgr::amd_comgr PROPERTIES
        IMPORTED_LOCATION "${amd_comgr_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${amd_comgr_INCLUDE_DIR}"
    )
endif()

if(TARGET amd_comgr::amd_comgr AND NOT TARGET amd_comgr)
    add_library(amd_comgr INTERFACE IMPORTED)
    target_link_libraries(amd_comgr INTERFACE amd_comgr::amd_comgr)
endif()

if(NOT amd_comgr_VERSION AND amd_comgr_LIBRARY)
    set(amd_comgr_VERSION "7.1.0")
endif()

find_package_handle_standard_args(amd_comgr
    REQUIRED_VARS amd_comgr_LIBRARY amd_comgr_INCLUDE_DIR
    VERSION_VAR amd_comgr_VERSION
)

mark_as_advanced(amd_comgr_INCLUDE_DIR amd_comgr_LIBRARY)
