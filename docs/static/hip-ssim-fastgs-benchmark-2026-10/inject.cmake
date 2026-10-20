if(NOT DEFINED LFS_EXTERNAL_INTEGRATED_BENCH_DIR)
    message(FATAL_ERROR "Set LFS_EXTERNAL_INTEGRATED_BENCH_DIR to the external integrated benchmark folder")
endif()
get_property(_lfs_integrated_bench_added GLOBAL PROPERTY LFS_EXTERNAL_INTEGRATED_BENCH_ADDED)
if(NOT _lfs_integrated_bench_added)
    set_property(GLOBAL PROPERTY LFS_EXTERNAL_INTEGRATED_BENCH_ADDED TRUE)
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
        CALL include "${LFS_EXTERNAL_INTEGRATED_BENCH_DIR}/CMakeLists.txt")
endif()
