# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

#[=============================================================================[
WindowsROCm.cmake

Helpers for locating the Windows ROCm/HIP SDK and the ROCm Python wheel layout.
The Windows packages have used several root names and compiler layouts across
7.1 and 7.2.x, so keep probing in one place instead of hard-coding a single SDK.
]=============================================================================]

function(lfs_windows_rocm_collect_roots out_var)
    set(_priority_roots "")
    set(_discovered_roots "")

    foreach(_var LFS_ROCM_PATH ROCM_PATH HIP_PATH)
        if(DEFINED ${_var} AND NOT "${${_var}}" STREQUAL "")
            list(APPEND _priority_roots "${${_var}}")
        endif()
    endforeach()

    foreach(_env_var HIP_PATH ROCM_PATH HIP_PATH_722 HIP_PATH_721 HIP_PATH_72 HIP_PATH_71 HIP_PATH_70 HIP_PATH_64 HIP_PATH_62)
        if(DEFINED ENV{${_env_var}} AND NOT "$ENV{${_env_var}}" STREQUAL "")
            list(APPEND _priority_roots "$ENV{${_env_var}}")
        endif()
    endforeach()

    find_program(_lfs_python_for_rocm NAMES python python3)
    if(_lfs_python_for_rocm)
        execute_process(
            COMMAND "${_lfs_python_for_rocm}" -c "import importlib.util, pathlib; spec = importlib.util.find_spec('_rocm_sdk_core'); print(pathlib.Path(spec.origin).resolve().parent if spec and spec.origin else '')"
            OUTPUT_VARIABLE _python_rocm_root
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_python_rocm_root)
            list(APPEND _priority_roots "${_python_rocm_root}")
        endif()
    endif()

    list(APPEND _discovered_roots
        "C:/Program Files/AMD/ROCm/7.2.2"
        "C:/Program Files/AMD/ROCm/7.2.1"
        "C:/Program Files/AMD/ROCm/7.2"
        "C:/Program Files/AMD/ROCm/7.1"
    )

    file(GLOB _program_files_roots LIST_DIRECTORIES true "C:/Program Files/AMD/ROCm/*")
    if(_program_files_roots)
        list(SORT _program_files_roots COMPARE NATURAL ORDER DESCENDING)
        list(APPEND _discovered_roots ${_program_files_roots})
    endif()

    set(_roots "")
    foreach(_root IN LISTS _priority_roots _discovered_roots)
        if(NOT _root STREQUAL "")
            file(TO_CMAKE_PATH "${_root}" _root_cmake)
            list(APPEND _roots "${_root_cmake}")
        endif()
    endforeach()
    if(_roots)
        list(REMOVE_DUPLICATES _roots)
    endif()

    set(${out_var} "${_roots}" PARENT_SCOPE)
endfunction()

function(lfs_windows_rocm_root_is_usable root out_var)
    set(_usable FALSE)
    if(EXISTS "${root}")
        if(EXISTS "${root}/include/hip/hip_runtime.h"
           OR EXISTS "${root}/lib/amdhip64.lib"
           OR EXISTS "${root}/bin/amdhip64.dll"
           OR EXISTS "${root}/bin/hipInfo.exe"
           OR EXISTS "${root}/bin/hipinfo.exe")
            set(_usable TRUE)
        endif()
    endif()
    set(${out_var} ${_usable} PARENT_SCOPE)
endfunction()

function(lfs_windows_rocm_find_root out_var)
    lfs_windows_rocm_collect_roots(_roots)
    set(_found "")
    foreach(_root IN LISTS _roots)
        lfs_windows_rocm_root_is_usable("${_root}" _usable)
        if(_usable)
            set(_found "${_root}")
            break()
        endif()
    endforeach()
    set(${out_var} "${_found}" PARENT_SCOPE)
endfunction()

function(lfs_windows_rocm_find_devel_root root out_var)
    set(_candidates "")

    foreach(_var LFS_ROCM_DEVEL_PATH ROCM_DEVEL_PATH HIP_DEVEL_PATH)
        if(DEFINED ${_var} AND NOT "${${_var}}" STREQUAL "")
            list(APPEND _candidates "${${_var}}")
        endif()
    endforeach()

    foreach(_env_var LFS_ROCM_DEVEL_PATH ROCM_DEVEL_PATH HIP_DEVEL_PATH)
        if(DEFINED ENV{${_env_var}} AND NOT "$ENV{${_env_var}}" STREQUAL "")
            list(APPEND _candidates "$ENV{${_env_var}}")
        endif()
    endforeach()

    find_program(_lfs_python_for_rocm_devel NAMES python python3)
    if(_lfs_python_for_rocm_devel)
        execute_process(
            COMMAND "${_lfs_python_for_rocm_devel}" -c "import importlib.util, pathlib; spec = importlib.util.find_spec('rocm_sdk_devel'); print(pathlib.Path(spec.origin).resolve().parent if spec and spec.origin else '')"
            OUTPUT_VARIABLE _python_rocm_devel_root
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_python_rocm_devel_root)
            list(APPEND _candidates "${_python_rocm_devel_root}")
        endif()
    endif()

    if(root)
        get_filename_component(_root_parent "${root}" DIRECTORY)
        list(APPEND _candidates
            "${root}"
            "${root}/devel"
            "${_root_parent}/rocm_sdk_devel")
    endif()

    set(_found "")
    foreach(_candidate IN LISTS _candidates)
        if(NOT _candidate STREQUAL "")
            file(TO_CMAKE_PATH "${_candidate}" _candidate_cmake)
            if(EXISTS "${_candidate_cmake}/include/hipcub/hipcub.hpp"
               OR EXISTS "${_candidate_cmake}/include/rocprim/rocprim.hpp"
               OR EXISTS "${_candidate_cmake}/include/thrust/complex.h")
                set(_found "${_candidate_cmake}")
                break()
            endif()
        endif()
    endforeach()

    set(${out_var} "${_found}" PARENT_SCOPE)
endfunction()

function(lfs_windows_rocm_find_runtime_dirs root out_var)
    set(_candidates "")

    foreach(_var LFS_ROCM_RUNTIME_PATH ROCM_RUNTIME_PATH HIP_RUNTIME_PATH)
        if(DEFINED ${_var} AND NOT "${${_var}}" STREQUAL "")
            list(APPEND _candidates "${${_var}}")
        endif()
    endforeach()

    foreach(_env_var LFS_ROCM_RUNTIME_PATH ROCM_RUNTIME_PATH HIP_RUNTIME_PATH)
        if(DEFINED ENV{${_env_var}} AND NOT "$ENV{${_env_var}}" STREQUAL "")
            list(APPEND _candidates "$ENV{${_env_var}}")
        endif()
    endforeach()

    if(root)
        get_filename_component(_root_parent "${root}" DIRECTORY)
        list(APPEND _candidates "${root}/bin")
        file(GLOB _wheel_runtime_bins LIST_DIRECTORIES true
            "${_root_parent}/_rocm_sdk_libraries_*/bin"
            "${_root_parent}/rocm_sdk_libraries_*/bin")
        if(_wheel_runtime_bins)
            list(APPEND _candidates ${_wheel_runtime_bins})
        endif()
    endif()

    set(_runtime_dirs "")
    foreach(_candidate IN LISTS _candidates)
        if(NOT _candidate STREQUAL "")
            file(TO_CMAKE_PATH "${_candidate}" _candidate_cmake)
            file(GLOB _candidate_dlls "${_candidate_cmake}/*.dll")
            if(_candidate_dlls)
                list(APPEND _runtime_dirs "${_candidate_cmake}")
            endif()
        endif()
    endforeach()

    if(_runtime_dirs)
        list(REMOVE_DUPLICATES _runtime_dirs)
    endif()
    set(${out_var} "${_runtime_dirs}" PARENT_SCOPE)
endfunction()

function(lfs_windows_rocm_find_clang root out_var)
    set(_clang "")
    foreach(_candidate
        "${root}/lib/llvm/bin/clang++.exe"
        "${root}/llvm/bin/clang++.exe"
        "${root}/bin/clang++.exe"
        "${root}/bin/hipcc.exe"
        "${root}/bin/hipcc.bat")
        if(EXISTS "${_candidate}")
            set(_clang "${_candidate}")
            break()
        endif()
    endforeach()
    set(${out_var} "${_clang}" PARENT_SCOPE)
endfunction()

function(lfs_windows_rocm_get_short_path root out_var)
    set(_short "")
    if(root)
        string(REPLACE "/" "\\" _native "${root}")
        execute_process(
            COMMAND cmd /c for %I in ("${_native}") do @echo %~sI
            OUTPUT_VARIABLE _short
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    if(_short STREQUAL "")
        set(_short "${root}")
    endif()
    file(TO_CMAKE_PATH "${_short}" _short_cmake)
    set(${out_var} "${_short_cmake}" PARENT_SCOPE)
endfunction()

function(lfs_windows_rocm_read_hip_version root out_version out_major out_minor out_patch)
    set(_major "")
    set(_minor "")
    set(_patch "")
    set(_version_header "${root}/include/hip/hip_version.h")

    if(EXISTS "${_version_header}")
        file(STRINGS "${_version_header}" _version_lines REGEX "#define HIP_VERSION_(MAJOR|MINOR|PATCH)[ \t]+[0-9]+")
        foreach(_line IN LISTS _version_lines)
            if(_line MATCHES "#define HIP_VERSION_MAJOR[ \t]+([0-9]+)")
                set(_major "${CMAKE_MATCH_1}")
            elseif(_line MATCHES "#define HIP_VERSION_MINOR[ \t]+([0-9]+)")
                set(_minor "${CMAKE_MATCH_1}")
            elseif(_line MATCHES "#define HIP_VERSION_PATCH[ \t]+([0-9]+)")
                set(_patch "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endif()

    if(_major STREQUAL "" OR _minor STREQUAL "")
        get_filename_component(_root_name "${root}" NAME)
        if(_root_name MATCHES "^([0-9]+)\\.([0-9]+)(\\.([0-9]+))?")
            set(_major "${CMAKE_MATCH_1}")
            set(_minor "${CMAKE_MATCH_2}")
            if(CMAKE_MATCH_4)
                set(_patch "${CMAKE_MATCH_4}")
            endif()
        endif()
    endif()

    if(_major STREQUAL "")
        set(_major "7")
    endif()
    if(_minor STREQUAL "")
        set(_minor "2")
    endif()
    if(_patch STREQUAL "")
        set(_patch "0")
    endif()

    set(${out_version} "${_major}.${_minor}.${_patch}" PARENT_SCOPE)
    set(${out_major} "${_major}" PARENT_SCOPE)
    set(${out_minor} "${_minor}" PARENT_SCOPE)
    set(${out_patch} "${_patch}" PARENT_SCOPE)
endfunction()

function(lfs_windows_rocm_find_device_lib_path root out_var)
    set(_device_lib_path "")
    foreach(_candidate
        "${root}/amdgcn/bitcode"
        "${root}/lib/llvm/amdgcn/bitcode"
        "${root}/lib/amdgcn/bitcode")
        if(EXISTS "${_candidate}")
            set(_device_lib_path "${_candidate}")
            break()
        endif()
    endforeach()

    if(_device_lib_path STREQUAL "")
        file(GLOB _clang_bitcode_roots LIST_DIRECTORIES true "${root}/lib/clang/*/amdgcn/bitcode")
        if(_clang_bitcode_roots)
            list(SORT _clang_bitcode_roots COMPARE NATURAL ORDER DESCENDING)
            list(GET _clang_bitcode_roots 0 _device_lib_path)
        endif()
    endif()

    if(_device_lib_path STREQUAL "")
        set(_device_lib_path "${root}/amdgcn/bitcode")
    endif()

    file(TO_CMAKE_PATH "${_device_lib_path}" _device_lib_path)
    set(${out_var} "${_device_lib_path}" PARENT_SCOPE)
endfunction()
