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

# Netlib CBLAS
add_executable(main_netlib ${CPU_FILE})
target_link_libraries(main_netlib PUBLIC /usr/lib/x86_64-linux-gnu/blas/libblas.so)
target_compile_definitions(main_netlib PRIVATE USE_NETLIB)

# cuBlas
add_executable(main_cu main.cu)
