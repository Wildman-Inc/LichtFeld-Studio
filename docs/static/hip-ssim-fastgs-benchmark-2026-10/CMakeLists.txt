cmake_minimum_required(VERSION 3.24)

add_executable(lfs_external_integrated_bench "${CMAKE_CURRENT_LIST_DIR}/integrated_bench.cu")
set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/integrated_bench.cu" PROPERTIES LANGUAGE HIP)
target_compile_features(lfs_external_integrated_bench PRIVATE cxx_std_23)
target_compile_definitions(lfs_external_integrated_bench PRIVATE
    USE_HIP=1 LFS_USE_HIP=1 USE_ROCM=1 __HIP_PLATFORM_AMD__
    LFS_BENCH_HAS_NATIVE_ATOMICS=$<BOOL:${LFS_BENCH_HAS_NATIVE_ATOMICS}>)
target_include_directories(lfs_external_integrated_bench PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/training/rasterization/fastgs/rasterization/include"
    "${CMAKE_SOURCE_DIR}/src/training/include"
    "${CMAKE_SOURCE_DIR}/src/hip_compat"
    "${CMAKE_SOURCE_DIR}/include/cuda_shim")
target_link_libraries(lfs_external_integrated_bench PRIVATE lfs_training lfs_core fastlfs_backend)
set_target_properties(lfs_external_integrated_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/integrated_bench/$<CONFIG>")
