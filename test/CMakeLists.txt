find_package(MKL CONFIG REQUIRED PATHS "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl")
add_executable(main_hw4 main_hw4.c)
target_link_libraries(main_hw4 PUBLIC MKL::MKL)

find_package(MKL CONFIG REQUIRED PATHS "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl")
find_package(OpenMP REQUIRED)
add_executable(main_hw4_omp main_hw4.c)
target_link_libraries(main_hw4_omp PUBLIC MKL::MKL OpenMP::OpenMP_C)
target_compile_definitions(main_hw4_omp PRIVATE USE_OMP)