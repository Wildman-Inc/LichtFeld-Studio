# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

#[=============================================================================[
 FindHIP.cmake - CMake module to find AMD ROCm/HIP

 This module defines the following variables:
   HIP_FOUND          - True if HIP is found
   HIP_VERSION        - HIP version string
   HIP_INCLUDE_DIRS   - Include directories for HIP
   HIP_LIBRARIES      - Libraries to link against
   HIP_HIPCC_EXECUTABLE - Path to hipcc compiler
   HIP_HIP_ARCH       - Target architecture flags

 And the following imported targets:
   hip::host          - HIP host library
   hip::device        - HIP device library
   roc::hipblas       - hipBLAS library
   roc::hiprand       - hipRAND library
   roc::hipcub        - hipCUB library (header-only)

 Additionally provides:
   hip_add_library()  - Function to create HIP libraries (Windows-compatible)
]=============================================================================]

# NOTE: This module only provides library discovery and hip_add_library() function.
# It does NOT create hip::host, hip::device, or hip::amdhip64 targets.
# On Windows, those targets should be created by find_package(hip CONFIG) from AMD ROCm SDK
# BEFORE find_package(Torch), because PyTorch's LoadHIP.cmake also calls find_package(hip CONFIG)
# and will conflict if targets are partially defined.

# Skip CONFIG mode here - let the main CMakeLists.txt handle it
# to ensure proper target creation order

# Try to locate HIP manually (for MODULE mode searches)
if(NOT HIP_FOUND AND NOT hip_FOUND)
    # Common ROCm installation paths
    set(ROCM_SEARCH_PATHS
        /opt/rocm
        /opt/rocm-7.0.0
        /opt/rocm-7.1.0
        $ENV{ROCM_PATH}
        $ENV{HIP_PATH}
    )

    # Find hipcc compiler
    find_program(HIP_HIPCC_EXECUTABLE
        NAMES hipcc
        PATHS ${ROCM_SEARCH_PATHS}
        PATH_SUFFIXES bin
    )

    # Find HIP include directory
    find_path(HIP_INCLUDE_DIR
        NAMES hip/hip_runtime.h
        PATHS ${ROCM_SEARCH_PATHS}
        PATH_SUFFIXES include
    )

    # Find HIP library
    find_library(HIP_LIBRARY
        NAMES amdhip64 hip_hcc
        PATHS ${ROCM_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64
    )

    # Get HIP version
    if(HIP_HIPCC_EXECUTABLE)
        execute_process(
            COMMAND ${HIP_HIPCC_EXECUTABLE} --version
            OUTPUT_VARIABLE HIP_VERSION_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(HIP_VERSION_OUTPUT MATCHES "HIP version: ([0-9]+\\.[0-9]+\\.[0-9]+)")
            set(HIP_VERSION ${CMAKE_MATCH_1})
        endif()
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(HIP
        REQUIRED_VARS HIP_HIPCC_EXECUTABLE HIP_INCLUDE_DIR HIP_LIBRARY
        VERSION_VAR HIP_VERSION
    )

    if(HIP_FOUND)
        set(HIP_INCLUDE_DIRS ${HIP_INCLUDE_DIR})
        set(HIP_LIBRARIES ${HIP_LIBRARY})

        # NOTE: Do NOT create hip::host, hip::device, or hip::amdhip64 targets here.
        # They should be created by find_package(hip CONFIG) to avoid conflicts with:
        # 1. AMD ROCm's hip-targets.cmake
        # 2. PyTorch's LoadHIP.cmake which calls find_package(hip CONFIG)
        #
        # The main CMakeLists.txt should call find_package(hip CONFIG) before find_package(Torch)
        # to ensure proper target creation and avoid "targets already defined" errors.
    endif()
endif()

# Find additional ROCm libraries
if(HIP_FOUND OR hip_FOUND)
    set(HIP_FOUND TRUE)

    # Get ROCm root from HIP location
    if(HIP_HIPCC_EXECUTABLE)
        get_filename_component(ROCM_ROOT "${HIP_HIPCC_EXECUTABLE}" DIRECTORY)
        get_filename_component(ROCM_ROOT "${ROCM_ROOT}" DIRECTORY)
    elseif(hip_DIR)
        get_filename_component(ROCM_ROOT "${hip_DIR}" DIRECTORY)
        get_filename_component(ROCM_ROOT "${ROCM_ROOT}" DIRECTORY)
        get_filename_component(ROCM_ROOT "${ROCM_ROOT}" DIRECTORY)
    else()
        set(ROCM_ROOT "/opt/rocm")
    endif()

    if(WIN32)
        # On Windows, just find the libraries without creating targets
        # PyTorch's LoadHIP.cmake will create the proper targets later
        # Creating targets here causes conflicts with PyTorch's find_package calls
        find_library(HIPBLAS_LIBRARY hipblas PATHS "${ROCM_ROOT}" PATH_SUFFIXES lib)
        find_library(HIPRAND_LIBRARY hiprand PATHS "${ROCM_ROOT}" PATH_SUFFIXES lib)
        find_path(HIPBLAS_INCLUDE_DIR hipblas/hipblas.h PATHS "${ROCM_ROOT}" PATH_SUFFIXES include)
        find_path(HIPRAND_INCLUDE_DIR hiprand/hiprand.h PATHS "${ROCM_ROOT}" PATH_SUFFIXES include)
        find_path(HIPCUB_INCLUDE_DIR hipcub/hipcub.hpp PATHS "${ROCM_ROOT}" PATH_SUFFIXES include)
        find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp PATHS "${ROCM_ROOT}" PATH_SUFFIXES include)
        find_path(ROCTHRUST_INCLUDE_DIR thrust/device_vector.h PATHS "${ROCM_ROOT}" PATH_SUFFIXES include)
        
        if(HIPBLAS_LIBRARY)
            set(hipblas_FOUND TRUE)
        endif()
        if(HIPRAND_LIBRARY)
            set(hiprand_FOUND TRUE)
        endif()
        if(HIPCUB_INCLUDE_DIR)
            set(hipcub_FOUND TRUE)
        endif()
        if(ROCPRIM_INCLUDE_DIR)
            set(rocprim_FOUND TRUE)
        endif()
        if(ROCTHRUST_INCLUDE_DIR)
            set(rocthrust_FOUND TRUE)
        endif()
    else()
        # Linux: use CONFIG mode and create targets if needed
        
        # Find hipBLAS
        find_package(hipblas CONFIG QUIET PATHS "${ROCM_ROOT}")
        if(NOT hipblas_FOUND)
            find_library(HIPBLAS_LIBRARY NAMES hipblas PATHS "${ROCM_ROOT}" PATH_SUFFIXES lib lib64)
            find_path(HIPBLAS_INCLUDE_DIR NAMES hipblas/hipblas.h PATHS "${ROCM_ROOT}" PATH_SUFFIXES include)
            if(HIPBLAS_LIBRARY AND HIPBLAS_INCLUDE_DIR)
                set(hipblas_FOUND TRUE)
                if(NOT TARGET roc::hipblas)
                    add_library(roc::hipblas SHARED IMPORTED)
                    set_target_properties(roc::hipblas PROPERTIES
                        IMPORTED_LOCATION "${HIPBLAS_LIBRARY}"
                        INTERFACE_INCLUDE_DIRECTORIES "${HIPBLAS_INCLUDE_DIR}"
                    )
                endif()
            endif()
        endif()

        # Find hipRAND
        find_package(hiprand CONFIG QUIET PATHS "${ROCM_ROOT}")
        if(NOT hiprand_FOUND)
            find_library(HIPRAND_LIBRARY NAMES hiprand PATHS "${ROCM_ROOT}" PATH_SUFFIXES lib lib64)
            find_path(HIPRAND_INCLUDE_DIR NAMES hiprand/hiprand.h PATHS "${ROCM_ROOT}" PATH_SUFFIXES include)
            if(HIPRAND_LIBRARY AND HIPRAND_INCLUDE_DIR)
                set(hiprand_FOUND TRUE)
                if(NOT TARGET roc::hiprand)
                    add_library(roc::hiprand SHARED IMPORTED)
                    set_target_properties(roc::hiprand PROPERTIES
                        IMPORTED_LOCATION "${HIPRAND_LIBRARY}"
                        INTERFACE_INCLUDE_DIRECTORIES "${HIPRAND_INCLUDE_DIR}"
                    )
                endif()
            endif()
        endif()

        # Find hipCUB (header-only)
        find_package(hipcub CONFIG QUIET PATHS "${ROCM_ROOT}")
        if(NOT hipcub_FOUND)
            find_path(HIPCUB_INCLUDE_DIR NAMES hipcub/hipcub.hpp PATHS "${ROCM_ROOT}" PATH_SUFFIXES include)
            if(HIPCUB_INCLUDE_DIR)
                set(hipcub_FOUND TRUE)
                if(NOT TARGET roc::hipcub)
                    add_library(roc::hipcub INTERFACE IMPORTED)
                    set_target_properties(roc::hipcub PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${HIPCUB_INCLUDE_DIR}"
                    )
                endif()
            endif()
        endif()

        # Find rocPRIM (required by hipCUB)
        find_package(rocprim CONFIG QUIET PATHS "${ROCM_ROOT}")
        if(NOT rocprim_FOUND)
            find_path(ROCPRIM_INCLUDE_DIR NAMES rocprim/rocprim.hpp PATHS "${ROCM_ROOT}" PATH_SUFFIXES include)
            if(ROCPRIM_INCLUDE_DIR)
                set(rocprim_FOUND TRUE)
                if(NOT TARGET roc::rocprim)
                    add_library(roc::rocprim INTERFACE IMPORTED)
                    set_target_properties(roc::rocprim PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${ROCPRIM_INCLUDE_DIR}"
                    )
                endif()
            endif()
        endif()

        # Find rocThrust
        find_package(rocthrust CONFIG QUIET PATHS "${ROCM_ROOT}")
        if(NOT rocthrust_FOUND)
            find_path(ROCTHRUST_INCLUDE_DIR NAMES thrust/device_vector.h PATHS "${ROCM_ROOT}" PATH_SUFFIXES include)
            if(ROCTHRUST_INCLUDE_DIR)
                set(rocthrust_FOUND TRUE)
                if(NOT TARGET roc::rocthrust)
                    add_library(roc::rocthrust INTERFACE IMPORTED)
                    set_target_properties(roc::rocthrust PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${ROCTHRUST_INCLUDE_DIR}"
                    )
                endif()
            endif()
        endif()
    endif()

    # Set HIP architecture flags
    # Common AMD GPU architectures:
    # gfx906  - MI50/MI60 (Vega 20)
    # gfx908  - MI100 (Arcturus)
    # gfx90a  - MI200 series (Aldebaran)
    # gfx940  - MI300X
    # gfx942  - MI300A
    # gfx1030 - RX 6800/6900 (RDNA 2)
    # gfx1100 - RX 7900 (RDNA 3)
    # gfx1101 - RX 7600/7700 (RDNA 3)
    # gfx1102 - RX 7600 XT (RDNA 3)

    if(NOT DEFINED HIP_ARCHITECTURES)
        # Default to common architectures for ROCm 7
        # MI300 series (datacenter), MI200 series, and RDNA 3 (consumer)
        set(HIP_ARCHITECTURES "gfx90a;gfx940;gfx942;gfx1100;gfx1101" CACHE STRING "HIP GPU architectures")
    endif()

    # Convert to hipcc flags
    set(HIP_HIP_ARCH "")
    foreach(arch ${HIP_ARCHITECTURES})
        list(APPEND HIP_HIP_ARCH "--offload-arch=${arch}")
    endforeach()
    string(REPLACE ";" " " HIP_HIP_ARCH "${HIP_HIP_ARCH}")

    message(STATUS "HIP found: ${HIP_VERSION}")
    message(STATUS "HIP compiler: ${HIP_HIPCC_EXECUTABLE}")
    message(STATUS "HIP architectures: ${HIP_ARCHITECTURES}")
    if(hipblas_FOUND)
        message(STATUS "hipBLAS: Found")
    endif()
    if(hiprand_FOUND)
        message(STATUS "hipRAND: Found")
    endif()
    if(hipcub_FOUND)
        message(STATUS "hipCUB: Found")
    endif()
    if(rocprim_FOUND)
        message(STATUS "rocPRIM: Found")
    endif()
    if(rocthrust_FOUND)
        message(STATUS "rocThrust: Found")
    endif()
endif()

mark_as_advanced(
    HIP_HIPCC_EXECUTABLE
    HIP_INCLUDE_DIR
    HIP_LIBRARY
    HIPBLAS_LIBRARY
    HIPBLAS_INCLUDE_DIR
    HIPRAND_LIBRARY
    HIPRAND_INCLUDE_DIR
    HIPCUB_INCLUDE_DIR
    ROCPRIM_INCLUDE_DIR
    ROCTHRUST_INCLUDE_DIR
)

#[=============================================================================[
 hip_add_library - Create a library with HIP source files

 This is a Windows-compatible implementation that doesn't rely on AMD's
 FindHIP.cmake, which has path escaping issues on Windows.

 Usage:
   hip_add_library(<name> STATIC|SHARED|MODULE <source1> [<source2>...])

 Variables that affect compilation:
   HIP_HIPCC_FLAGS - Additional flags for hipcc/clang++
   HIP_CLANG_PATH  - Path to clang++ (8.3 short path recommended on Windows)
]=============================================================================]
function(hip_add_library target_name lib_type)
    set(sources ${ARGN})
    
    # Add library
    add_library(${target_name} ${lib_type} ${sources})
    
    if(WIN32)
        # On Windows, use add_custom_command to compile HIP sources
        # because CMake doesn't natively support HIP language on Windows
        
        get_target_property(target_sources ${target_name} SOURCES)
        
        # Get HIP compiler path with short path format
        if(DEFINED HIP_CLANG_PATH)
            set(HIP_CXX "${HIP_CLANG_PATH}/clang++.exe")
        elseif(DEFINED ENV{HIP_PATH})
            set(HIP_CXX "$ENV{HIP_PATH}/llvm/bin/clang++.exe")
        else()
            message(FATAL_ERROR "HIP_CLANG_PATH not set for Windows HIP compilation")
        endif()
        
        # Set source file properties for HIP compilation
        foreach(source ${sources})
            get_filename_component(source_ext ${source} EXT)
            if(source_ext STREQUAL ".cu" OR source_ext STREQUAL ".hip")
                # Mark as HIP source for custom compilation
                set_source_files_properties(${source} PROPERTIES
                    LANGUAGE CXX
                )
            endif()
        endforeach()
        
        # Set target properties for HIP compilation
        set_target_properties(${target_name} PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )
        
        # Add HIP compile definitions
        target_compile_definitions(${target_name} PRIVATE
            __HIP_PLATFORM_AMD__
            USE_ROCM
        )
        
        # Build architecture flags
        if(DEFINED HIP_ARCHITECTURES)
            set(arch_flags "")
            foreach(arch ${HIP_ARCHITECTURES})
                list(APPEND arch_flags "--offload-arch=${arch}")
            endforeach()
        endif()
        
        # Set compile options for HIP files
        # Use generator expression to apply only to .cu and .hip files
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-x$<SEMICOLON>hip>
            $<$<COMPILE_LANGUAGE:CXX>:${arch_flags}>
        )
        
        # Parse HIP_HIPCC_FLAGS if set
        if(DEFINED HIP_HIPCC_FLAGS)
            separate_arguments(hipcc_flags NATIVE_COMMAND "${HIP_HIPCC_FLAGS}")
            target_compile_options(${target_name} PRIVATE ${hipcc_flags})
        endif()
        
    else()
        # On Linux, use CMake's native HIP language support
        set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
        set_target_properties(${target_name} PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
        )
    endif()
endfunction()
