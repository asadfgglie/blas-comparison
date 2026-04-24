find_package(MKL CONFIG REQUIRED PATHS "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl")
add_executable(main_hw4 main_hw4.c)
target_link_libraries(main_hw4 PUBLIC MKL::MKL)