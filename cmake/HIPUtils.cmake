# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

#[=============================================================================[
 HIPUtils.cmake - Windows-compatible HIP compilation utilities

 This module provides a lfs_hip_add_library() function that doesn't rely on
 AMD's FindHIP.cmake, which has Windows path escaping issues.

 We use a unique name (lfs_hip_add_library) instead of hip_add_library to
 avoid conflicts with AMD's HIP_ADD_LIBRARY macro from FindHIP.cmake.
]=============================================================================]

# Use a unique name to avoid conflicts with AMD's HIP_ADD_LIBRARY
macro(lfs_hip_add_library target_name lib_type)
    set(_sources ${ARGN})
    
    if(WIN32)
        # On Windows, create a regular C++ library
        # We'll rename .cu files to .hip and compile them directly with clang
        # since CMake's HIP language support is limited on Windows
        
        # First, copy .cu files to .hip in the binary directory
        set(_hip_sources "")
        foreach(_source ${_sources})
            get_filename_component(_source_ext ${_source} EXT)
            get_filename_component(_source_name ${_source} NAME_WE)
            get_filename_component(_source_dir ${_source} DIRECTORY)
            
            if(_source_ext STREQUAL ".cu")
                # Get absolute path for the source
                if(IS_ABSOLUTE ${_source})
                    set(_source_abs ${_source})
                else()
                    set(_source_abs "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
                endif()
                
                # Create .hip symlink/copy in binary dir
                set(_hip_file "${CMAKE_CURRENT_BINARY_DIR}/${_source_name}.hip.cpp")
                
                # Configure a simple wrapper file that includes the original
                # Include hip_runtime_compat.h first to block cuda_runtime.h and provide CUDA->HIP mappings
                # Then include hip_runtime.h for HIP intrinsics
                file(WRITE "${_hip_file}" 
                    "// Auto-generated HIP wrapper for ${_source}\n"
                    "#include \"core/cuda/hip_runtime_compat.h\"\n"
                    "#include <hip/hip_runtime.h>\n"
                    "#include \"${_source_abs}\"\n"
                )
                
                list(APPEND _hip_sources "${_hip_file}")
            else()
                # Keep non-CUDA sources as-is
                list(APPEND _hip_sources "${_source}")
            endif()
        endforeach()
        
        add_library(${target_name} ${lib_type} ${_hip_sources})
        
        # Set target properties for clang compilation
        set_target_properties(${target_name} PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
            # Disable C++ scan for these files to avoid clang-scan-deps issues
            CXX_SCAN_FOR_MODULES OFF
        )
        
        # Add HIP compile definitions
        target_compile_definitions(${target_name} PRIVATE
            __HIP_PLATFORM_AMD__
            USE_ROCM
        )
        
        # Build architecture flags - exclude gfx940 (not supported on Windows ROCm 7.10)
        set(_arch_flags "")
        if(DEFINED HIP_ARCHITECTURES)
            foreach(_arch ${HIP_ARCHITECTURES})
                # Skip unsupported architectures
                if(NOT _arch STREQUAL "gfx940")
                    list(APPEND _arch_flags "--offload-arch=${_arch}")
                endif()
            endforeach()
        endif()
        
        # Add HIP compile options directly - without -x c++ flag
        # Use COMPILE_OPTIONS to add flags that work with clang for HIP
        target_compile_options(${target_name} PRIVATE
            # Tell clang this is HIP code
            "SHELL:-x hip"
            # GPU architectures
            ${_arch_flags}
            # Disable OpenMP to avoid conflicts between clang's openmp_wrappers headers and MSVC STL
            # The openmp_wrappers/math.h causes type_traits to have vectorcall redefinition errors
            -fno-openmp
        )
        
        # Parse HIP_HIPCC_FLAGS if set
        if(DEFINED HIP_HIPCC_FLAGS)
            separate_arguments(_hipcc_flags NATIVE_COMMAND "${HIP_HIPCC_FLAGS}")
            target_compile_options(${target_name} PRIVATE ${_hipcc_flags})
        endif()
        
        # Add HIP include directories
        if(DEFINED HIP_INCLUDE_DIRS)
            target_include_directories(${target_name} PRIVATE ${HIP_INCLUDE_DIRS})
        endif()
        
        message(STATUS "HIPUtils: Created ${target_name} with Windows-compatible HIP compilation")
        
    else()
        # On Linux, use CMake's native HIP language support
        add_library(${target_name} ${lib_type} ${_sources})

        # Compile only GPU translation units as HIP; keep regular C++ sources as CXX.
        set(_hip_language_sources "")
        foreach(_source ${_sources})
            get_filename_component(_source_ext ${_source} EXT)
            if(_source_ext STREQUAL ".cu" OR _source_ext STREQUAL ".hip" OR _source_ext STREQUAL ".hip.cpp")
                list(APPEND _hip_language_sources ${_source})
            endif()
        endforeach()
        if(_hip_language_sources)
            set_source_files_properties(${_hip_language_sources} PROPERTIES LANGUAGE HIP)
        endif()

        set_target_properties(${target_name} PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
        )
        
        target_compile_definitions(${target_name} PRIVATE
            __HIP_PLATFORM_AMD__
            USE_ROCM
        )
        
        # Build architecture flags for HIP
        if(DEFINED HIP_ARCHITECTURES)
            set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${HIP_ARCHITECTURES})
        endif()
    endif()
endmacro()

# Wrapper for cross-platform compatibility
# This allows existing code using hip_add_library to work with our implementation
macro(hip_add_library target_name lib_type)
    lfs_hip_add_library(${target_name} ${lib_type} ${ARGN})
endmacro()

message(STATUS "HIPUtils.cmake loaded - lfs_hip_add_library() available")
