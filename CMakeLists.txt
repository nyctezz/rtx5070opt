cmake_minimum_required(VERSION 3.20)
project(blackwell_tensor_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files (updated to .cu so nvcc compiles both)
set(SOURCES
    src/01_Kernel.cu
    src/02_Kernel.cu
    benchmarks/benchmark_matrix_multiplication.cu
)

add_executable(tensor_bench ${SOURCES})

# Automatically detect your RTX 5070 and compile for its exact architecture
set_property(TARGET tensor_bench PROPERTY CUDA_ARCHITECTURES native)