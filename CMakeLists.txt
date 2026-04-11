cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(blas-test LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 14)

set(CPU_FILE main.c utils.c utils.h)

include_directories(.)

# Intel oneMKL
find_package(MKL CONFIG REQUIRED PATHS "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl")
add_executable(main_oneMKL ${CPU_FILE})
target_link_libraries(main_oneMKL PUBLIC MKL::MKL)
target_compile_definitions(main_oneMKL PRIVATE USE_MKL)

# OpenBlas
add_executable(main_openblas ${CPU_FILE})
target_link_libraries(main_openblas PUBLIC openblas)
target_compile_definitions(main_openblas PRIVATE USE_OPENBLAS)

# single thread & process Intel oneMKL
add_executable(main_oneMKL_one_thread_process ${CPU_FILE})
target_include_directories(main_oneMKL_one_thread_process PUBLIC /opt/intel/oneapi/mkl/latest/include)
target_link_libraries(main_oneMKL_one_thread_process PUBLIC
    -Wl,--start-group
    /opt/intel/oneapi/mkl/latest/lib/libmkl_intel_ilp64.so
    /opt/intel/oneapi/mkl/latest/lib/libmkl_sequential.so
    /opt/intel/oneapi/mkl/latest/lib/libmkl_core.so
    -Wl,--end-group
    m pthread)
target_compile_definitions(main_oneMKL_one_thread_process PRIVATE USE_MKL)

# single thread & process OpenBLAS (link to pure serial version)
add_executable(main_openblas_one_thread_process ${CPU_FILE})
target_link_libraries(main_openblas_one_thread_process PUBLIC /usr/lib/x86_64-linux-gnu/openblas-serial/libopenblas.so)
target_compile_definitions(main_openblas_one_thread_process PRIVATE USE_OPENBLAS)

# Netlib CBLAS
add_executable(main_netlib ${CPU_FILE})
target_link_libraries(main_netlib PUBLIC /usr/lib/x86_64-linux-gnu/blas/libblas.so)
target_compile_definitions(main_netlib PRIVATE USE_NETLIB)

# cuBlas
add_executable(main_cu main.cu)
