#include <iostream>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>

#include "kernels.cuh"
#include "benchmarks.cuh"



void run_b01_k01()
{
    int n = 2048;
    std::string kernel_name = "kernel_01";
    dim3 block_size(16, 16);

    std::cout << "\n=-----------------------=\n";
    std::cout << "initializing matrices of size " << n << " x " << n << " with 1.0f\n";
    benchmark_01 bnch_01(n);
    bnch_01.init_matrices_with_ones();

    std::cout << "running: " << kernel_name << " with:\n";
    std::cout << "> block Size: (" << block_size.x << ", " << block_size.y << ")";
    std::cout << "\n>>>\n";


    benchmark_result result = bnch_01.run(kernel_01, block_size);


    std::cout << kernel_name << " time: " << result.avg_ms << " ms\n";
    std::cout << kernel_name << " performance: " << result.gflops << " GFLOPS\n";
    std::cout << kernel_name << " calculating bytes speed: " << result.achieved_compute_gb_s << " GB/s\n";
    std::cout << "=-----------------------=\n\n";
}

void run_b01_k02()
{
    int n = 2048;
    std::string kernel_name = "kernel_02";
    dim3 block_size(16, 16);

        std::cout << "\n=-----------------------=\n";
    std::cout << "initializing matrices of size " << n << " x " << n << " with 1.0f\n";
    benchmark_01 bnch_01(n);
    bnch_01.init_matrices_with_ones();

    std::cout << "running: " << kernel_name << " with:\n";
    std::cout << "> block Size: (" << block_size.x << ", " << block_size.y << ")";
    std::cout << "\n>>>\n";


    benchmark_result result = bnch_01.run(kernel_02, block_size);


    std::cout << kernel_name << " time: " << result.avg_ms << " ms\n";
    std::cout << kernel_name << " performance: " << result.gflops << " GFLOPS\n";
    std::cout << kernel_name << " calculating bytes speed: " << result.achieved_compute_gb_s << " GB/s\n";
    std::cout << "=-----------------------=\n\n";
}


int main() 
{
    run_b01_k01();
    run_b01_k02();

    return 0;
}