# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# MIOpen is optional on some Windows HIP SDK installations.
# PyTorch's LoadHIP.cmake still marks it REQUIRED, so we provide a best-effort
# finder that falls back to an interface target on Windows when no package exists.

include(FindPackageHandleStandardArgs)

set(_miopen_rocm_hints "")
if(DEFINED ROCM_PATH AND NOT "${ROCM_PATH}" STREQUAL "")
    list(APPEND _miopen_rocm_hints "${ROCM_PATH}")
endif()
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    list(APPEND _miopen_rocm_hints "$ENV{ROCM_PATH}")
endif()
if(DEFINED ENV{MIOPEN_PATH} AND NOT "$ENV{MIOPEN_PATH}" STREQUAL "")
    list(APPEND _miopen_rocm_hints "$ENV{MIOPEN_PATH}")
endif()
list(APPEND _miopen_rocm_hints "C:/opt/miopen" "C:/Program Files/AMD/ROCm/7.1" "C:/Program Files/AMD/ROCm/7.10")

find_path(miopen_INCLUDE_DIR
    NAMES miopen/miopen.h
    HINTS ${_miopen_rocm_hints}
    PATH_SUFFIXES include
)

find_library(miopen_LIBRARY
    NAMES MIOpen miopen
    HINTS ${_miopen_rocm_hints}
    PATH_SUFFIXES lib lib64
)

set(_miopen_stub FALSE)
if(miopen_LIBRARY)
    if(NOT TARGET MIOpen)
        add_library(MIOpen UNKNOWN IMPORTED)
        set_target_properties(MIOpen PROPERTIES
            IMPORTED_LOCATION "${miopen_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${miopen_INCLUDE_DIR}"
        )
    endif()
elseif(WIN32)
    # No MIOpen SDK installed: provide a no-op target so HIP consumers can link.
    set(_miopen_stub TRUE)
    if(NOT TARGET MIOpen)
        add_library(MIOpen INTERFACE IMPORTED)
    endif()
endif()

if(_miopen_stub)
    set(miopen_FOUND TRUE)
    set(miopen_VERSION "stub")
    if(NOT miopen_FIND_QUIETLY)
        message(WARNING "MIOpen library not found. Using stub INTERFACE target 'MIOpen' for Windows HIP build.")
    endif()
else()
    if(NOT miopen_VERSION AND miopen_LIBRARY)
        set(miopen_VERSION "1.0.0")
    endif()
    find_package_handle_standard_args(miopen
        REQUIRED_VARS miopen_LIBRARY
        VERSION_VAR miopen_VERSION
    )
endif()

mark_as_advanced(miopen_INCLUDE_DIR miopen_LIBRARY)
