# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

#[=============================================================================[
WindowsROCm.cmake

Helpers for locating the Windows ROCm/HIP SDK and the ROCm Python wheel layout.
The Windows packages have used several root names and compiler layouts across
ROCm 7.x and 10.x. Keep each selected SDK's headers and runtime together.
]=============================================================================]

function(lfs_windows_rocm_collect_roots out_var)
    set(_priority_roots "")
    set(_discovered_roots "")

    foreach(_var LFS_ROCM_PATH ROCM_PATH HIP_PATH)
        if(DEFINED ${_var} AND NOT "${${_var}}" STREQUAL "")
            list(APPEND _priority_roots "${${_var}}")
        endif()
    endforeach()

    foreach(_env_var LFS_ROCM_PATH ROCM_PATH HIP_PATH HIP_PATH_722 HIP_PATH_721 HIP_PATH_72 HIP_PATH_71 HIP_PATH_70 HIP_PATH_64 HIP_PATH_62)
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
           OR EXISTS "${root}/bin/amdhip64_7.dll"
           OR EXISTS "${root}/bin/hipInfo.exe"
           OR EXISTS "${root}/bin/hipinfo.exe")
            set(_usable TRUE)
        endif()
    endif()
    set(${out_var} ${_usable} PARENT_SCOPE)
endfunction()

function(lfs_windows_rocm_find_root out_var)
    # An explicit SDK must not silently fall back to another Python environment.
    foreach(_var LFS_ROCM_PATH ROCM_PATH HIP_PATH)
        if(DEFINED ${_var} AND NOT "${${_var}}" STREQUAL "")
            file(TO_CMAKE_PATH "${${_var}}" _explicit_root)
            lfs_windows_rocm_root_is_usable("${_explicit_root}" _usable)
            if(NOT _usable)
                message(FATAL_ERROR "${_var} does not contain a ROCm/HIP SDK: ${_explicit_root}")
            endif()
            set(${out_var} "${_explicit_root}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    foreach(_env_var LFS_ROCM_PATH ROCM_PATH HIP_PATH)
        if(DEFINED ENV{${_env_var}} AND NOT "$ENV{${_env_var}}" STREQUAL "")
            file(TO_CMAKE_PATH "$ENV{${_env_var}}" _explicit_root)
            lfs_windows_rocm_root_is_usable("${_explicit_root}" _usable)
            if(NOT _usable)
                message(FATAL_ERROR "${_env_var} does not contain a ROCm/HIP SDK: ${_explicit_root}")
            endif()
            set(${out_var} "${_explicit_root}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
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

    # Resolve relative to the chosen core, not `python` on PATH (which may be a
    # different environment, or vcpkg's Python after project() runs).
    if(root)
        get_filename_component(_root_parent "${root}" DIRECTORY)
        list(APPEND _candidates
            "${root}"
            "${root}/devel"
            "${_root_parent}/_rocm_sdk_devel"
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
        # ROCm 10.x uses one multi-architecture libraries wheel. Old per-device
        # wheels can remain installed after upgrading; do not mix their DLLs.
        set(_wheel_runtime_bins "")
        foreach(_libraries_root _rocm_sdk_libraries rocm_sdk_libraries)
            set(_runtime_bin "${_root_parent}/${_libraries_root}/bin")
            file(GLOB _runtime_dlls "${_runtime_bin}/*.dll")
            if(_runtime_dlls)
                list(APPEND _wheel_runtime_bins "${_runtime_bin}")
            endif()
        endforeach()
        if(NOT _wheel_runtime_bins)
            file(GLOB _wheel_runtime_bins LIST_DIRECTORIES true
                "${_root_parent}/_rocm_sdk_libraries_*/bin"
                "${_root_parent}/rocm_sdk_libraries_*/bin")
        endif()
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

function(lfs_windows_rocm_collect_runtime_files root out_var)
    lfs_windows_rocm_find_runtime_dirs("${root}" _runtime_dirs)
    lfs_windows_rocm_find_clang("${root}" _clang)
    get_filename_component(_compiler_dir "${_clang}" DIRECTORY)
    set(_readobj "${_compiler_dir}/llvm-readobj.exe")
    if(NOT EXISTS "${_readobj}")
        message(FATAL_ERROR "ROCm runtime dependency inspection requires ${_readobj}")
    endif()

    # Include the runtime compiler because HIP loads it dynamically. Follow PE
    # imports for everything else, limited to the selected SDK's directories.
    set(_pending "")
    set(_seed_names "")
    foreach(_dir IN LISTS _runtime_dirs)
        file(GLOB _seeds
            "${_dir}/amdhip64*.dll" "${_dir}/hiprtc*.dll"
            "${_dir}/hiprand.dll" "${_dir}/rocrand.dll")
        foreach(_seed IN LISTS _seeds)
            get_filename_component(_name "${_seed}" NAME)
            string(TOLOWER "${_name}" _name)
            if(NOT _name IN_LIST _seed_names)
                list(APPEND _seed_names "${_name}")
                list(APPEND _pending "${_seed}")
            endif()
        endforeach()
    endforeach()
    if(NOT _seed_names MATCHES "amdhip64.*[.]dll")
        message(FATAL_ERROR "No HIP runtime DLL found in ${_runtime_dirs}")
    endif()

    set(_files "")
    while(_pending)
        list(POP_FRONT _pending _dll)
        if(_dll IN_LIST _files)
            continue()
        endif()
        list(APPEND _files "${_dll}")
        execute_process(COMMAND "${_readobj}" --coff-imports "${_dll}"
            RESULT_VARIABLE _result OUTPUT_VARIABLE _imports ERROR_VARIABLE _error)
        if(NOT _result EQUAL 0)
            message(FATAL_ERROR "Cannot inspect ROCm runtime ${_dll}: ${_error}")
        endif()
        string(REGEX MATCHALL "Name: [^\r\n]+[.][dD][lL][lL]" _import_names "${_imports}")
        foreach(_import IN LISTS _import_names)
            string(REGEX REPLACE "^Name: " "" _name "${_import}")
            # NO_CACHE controls storage, not lookup: mask an inherited/cache
            # value so every import is resolved within this SDK's directories.
            set(_dependency "_dependency-NOTFOUND")
            find_file(_dependency NAMES "${_name}" PATHS ${_runtime_dirs}
                NO_DEFAULT_PATH NO_CACHE)
            if(_dependency)
                list(APPEND _pending "${_dependency}")
            else()
                string(TOLOWER "${_name}" _lower_name)
                if(_lower_name MATCHES "^(amdhip|amd_comgr|hip|roc)")
                    message(FATAL_ERROR
                        "${_dll} requires ${_name}, missing from the selected ROCm SDK")
                endif()
                # Windows and MSVC runtime DLLs are supplied by the OS and the
                # existing application runtime staging, not by another ROCm SDK.
            endif()
        endforeach()
    endwhile()
    set(${out_var} "${_files}" PARENT_SCOPE)
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

function(lfs_windows_rocm_collect_runtime_kpacks root architectures runtime_files out_var)
    lfs_windows_rocm_find_clang("${root}" _clang)
    get_filename_component(_compiler_dir "${_clang}" DIRECTORY)
    set(_kpacks "")
    foreach(_dll IN LISTS runtime_files)
        # PE section names are limited to eight characters. TheRock puts the
        # external device-code paths in the .kpackrf section on Windows.
        execute_process(
            COMMAND "${_compiler_dir}/llvm-readobj.exe" --string-dump=.kpackrf "${_dll}"
            RESULT_VARIABLE _result OUTPUT_VARIABLE _metadata ERROR_VARIABLE _error)
        if(NOT _result EQUAL 0)
            message(FATAL_ERROR "Cannot inspect ROCm device assets for ${_dll}: ${_error}")
        endif()
        string(REGEX MATCHALL "[.][.]/[.]kpack/[A-Za-z0-9_@.-]+[.]kpack" _patterns "${_metadata}")
        list(REMOVE_DUPLICATES _patterns)
        get_filename_component(_dll_dir "${_dll}" DIRECTORY)
        foreach(_pattern IN LISTS _patterns)
            if(_pattern MATCHES "@GFXARCH@")
                if(NOT architectures)
                    message(FATAL_ERROR "Select HIP_ARCHITECTURES to stage ROCm device assets")
                endif()
                set(_asset_architectures "${architectures}")
            else()
                set(_asset_architectures "all")
            endif()
            foreach(_arch IN LISTS _asset_architectures)
                string(REGEX REPLACE ":.*$" "" _arch "${_arch}")
                string(REPLACE "@GFXARCH@" "${_arch}" _relative_path "${_pattern}")
                get_filename_component(_asset "${_dll_dir}/${_relative_path}" ABSOLUTE)
                if(NOT EXISTS "${_asset}")
                    message(FATAL_ERROR
                        "Missing ROCm device archive ${_asset}. Install the matching "
                        "rocm-sdk-device-${_arch} wheel and run python -m rocm_sdk init.")
                endif()
                list(APPEND _kpacks "${_asset}")
            endforeach()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _kpacks)
    set(${out_var} "${_kpacks}" PARENT_SCOPE)
endfunction()

# Record the selected SDK inputs, including non-DLL device archives. This is an
# integrity contract for a local package, not a redistribution-license audit.
function(lfs_windows_rocm_install_artifacts root runtime_files kpacks)
    set(_sources ${runtime_files} ${kpacks})
    set(_destinations)
    foreach(_source IN LISTS runtime_files)
        get_filename_component(_name "${_source}" NAME)
        list(APPEND _destinations "${CMAKE_INSTALL_BINDIR}/${_name}")
    endforeach()
    foreach(_source IN LISTS kpacks)
        get_filename_component(_name "${_source}" NAME)
        list(APPEND _destinations ".kpack/${_name}")
    endforeach()
    file(GLOB_RECURSE _notices LIST_DIRECTORIES false "${root}/share/doc/*")
    foreach(_source IN LISTS _notices)
        file(RELATIVE_PATH _relative "${root}/share/doc" "${_source}")
        list(APPEND _sources "${_source}")
        list(APPEND _destinations "licenses/ROCm/runtime/${_relative}")
    endforeach()
    foreach(_provenance IN ITEMS "share/therock/therock_manifest.json" ".info/version")
        if(EXISTS "${root}/${_provenance}")
            get_filename_component(_name "${_provenance}" NAME)
            if(_name STREQUAL "version")
                set(_name "rocm-sdk-version.txt")
            endif()
            list(APPEND _sources "${root}/${_provenance}")
            list(APPEND _destinations "licenses/ROCm/provenance/${_name}")
        endif()
    endforeach()
    set(_manifest "")
    foreach(_source _destination IN ZIP_LISTS _sources _destinations)
        file(SHA256 "${_source}" _sha256)
        string(APPEND _manifest "${_destination}|${_sha256}\n")
        get_filename_component(_directory "${_destination}" DIRECTORY)
        get_filename_component(_name "${_destination}" NAME)
        install(FILES "${_source}" DESTINATION "${_directory}" RENAME "${_name}" COMPONENT runtime)
    endforeach()
    set(_manifest_file "${CMAKE_BINARY_DIR}/rocm-artifact-sha256-manifest.txt")
    file(CONFIGURE OUTPUT "${_manifest_file}" CONTENT "${_manifest}" @ONLY)
    install(FILES "${_manifest_file}" DESTINATION licenses/ROCm
        RENAME artifact-sha256-manifest.txt COMPONENT runtime)
endfunction()

# clang++ uses the MSVC ABI without setting CMake's MSVC frontend flag, so
# InstallRequiredSystemLibraries does not discover the Visual C++ CRT for it.
function(lfs_windows_rocm_find_msvc_runtime out_var)
    set(_redist_roots)
    if(NOT "$ENV{VCToolsRedistDir}" STREQUAL "")
        file(TO_CMAKE_PATH "$ENV{VCToolsRedistDir}" _explicit_redist)
        list(APPEND _redist_roots "${_explicit_redist}")
    endif()
    foreach(_include IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
        file(TO_CMAKE_PATH "${_include}" _include)
        if(_include MATCHES "^(.*)/[Vv][Cc]/[Tt]ools/[Mm][Ss][Vv][Cc]/([0-9.]+)/include(/.*)?$")
            set(_vs_root "${CMAKE_MATCH_1}")
            file(GLOB _compiler_redists LIST_DIRECTORIES true "${_vs_root}/VC/Redist/MSVC/[0-9]*")
            list(SORT _compiler_redists COMPARE NATURAL ORDER DESCENDING)
            list(APPEND _redist_roots ${_compiler_redists})
        endif()
    endforeach()
    # Sort installed fallbacks by runtime version, not VS directory name (2022
    # sorts after 18 even though VS 18 may carry the newer runtime).
    file(TO_CMAKE_PATH "$ENV{ProgramFiles}" _program_files)
    file(GLOB _installed_redists LIST_DIRECTORIES true
        "${_program_files}/Microsoft Visual Studio/*/*/VC/Redist/MSVC/[0-9]*")
    set(_versioned_redists)
    foreach(_redist IN LISTS _installed_redists)
        get_filename_component(_version "${_redist}" NAME)
        list(APPEND _versioned_redists "${_version}|${_redist}")
    endforeach()
    list(SORT _versioned_redists COMPARE NATURAL ORDER DESCENDING)
    foreach(_entry IN LISTS _versioned_redists)
        string(REGEX REPLACE "^[^|]+[|]" "" _redist "${_entry}")
        list(APPEND _redist_roots "${_redist}")
    endforeach()
    list(REMOVE_DUPLICATES _redist_roots)
    foreach(_redist IN LISTS _redist_roots)
        file(GLOB _crt_dirs LIST_DIRECTORIES true "${_redist}/x64/Microsoft.VC*.CRT")
        list(SORT _crt_dirs COMPARE NATURAL ORDER DESCENDING)
        foreach(_crt_dir IN LISTS _crt_dirs)
            if(EXISTS "${_crt_dir}/msvcp140.dll" AND EXISTS "${_crt_dir}/vcruntime140.dll")
                file(GLOB _runtimes LIST_DIRECTORIES false
                    "${_crt_dir}/msvcp140*.dll" "${_crt_dir}/vcruntime140*.dll"
                    "${_crt_dir}/concrt140.dll" "${_crt_dir}/vccorlib140.dll")
                list(FILTER _runtimes EXCLUDE REGEX "140d([_.]|[.]dll$)")
                message(STATUS "MSVC distributable runtime: ${_crt_dir}")
                set(${out_var} "${_runtimes}" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endforeach()
    message(FATAL_ERROR
        "MSVC x64 redistributable CRT was not found. Install Visual Studio C++ tools "
        "or set VCToolsRedistDir to its VC/Redist/MSVC/<version> directory.")
endfunction()

function(lfs_windows_find_sdk_tool tool_name out_var)
    set(_candidates "")

    if(DEFINED ENV{WindowsSdkDir} AND NOT "$ENV{WindowsSdkDir}" STREQUAL "")
        file(TO_CMAKE_PATH "$ENV{WindowsSdkDir}" _windows_sdk_dir)
        if(DEFINED ENV{WindowsSDKVersion} AND NOT "$ENV{WindowsSDKVersion}" STREQUAL "")
            string(REGEX REPLACE "[/\\]+$" "" _windows_sdk_version "$ENV{WindowsSDKVersion}")
            list(APPEND _candidates
                "${_windows_sdk_dir}/bin/${_windows_sdk_version}/x64/${tool_name}"
                "${_windows_sdk_dir}/bin/${_windows_sdk_version}/x86/${tool_name}")
        endif()
    endif()

    file(GLOB _sdk_x64_tools LIST_DIRECTORIES false
        "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/${tool_name}"
        "C:/Program Files (x86)/Windows Kits/11/bin/*/x64/${tool_name}")
    if(_sdk_x64_tools)
        list(SORT _sdk_x64_tools COMPARE NATURAL ORDER DESCENDING)
        list(APPEND _candidates ${_sdk_x64_tools})
    endif()

    find_program(_sdk_tool_from_path NAMES ${tool_name})
    if(_sdk_tool_from_path)
        list(APPEND _candidates "${_sdk_tool_from_path}")
    endif()

    set(_found "")
    foreach(_candidate IN LISTS _candidates)
        if(_candidate AND EXISTS "${_candidate}")
            set(_found "${_candidate}")
            break()
        endif()
    endforeach()

    set(${out_var} "${_found}" PARENT_SCOPE)
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
