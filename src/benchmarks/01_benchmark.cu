#include <iostream>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>

#include "kernels.cuh"


// Error reporter macro
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) 
{
    if (code != cudaSuccess) 
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) 
                  << " | File: " << file 
                  << " | Line: " << line << std::endl;
        if (abort) exit(code);
    }
}

#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/*
    01_benchmark - matrix multiplication benchmark:

this benchmark evaluates performance of a given kernel for matrix multiplication task

*/

int main() 
{
    


    int N = 2048;
    size_t bytes = N * N * sizeof(float);

    std::cout << "Initializing matrices of size " << N << "x" << N << "...\n";
    std::vector<float> h_A(N * N, 1.0f);
    std::vector<float> h_B(N * N, 1.0f);
    std::vector<float> h_C(N * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // double-precision floating-point operations count for matrix multiplication: 2 * N^3
    double ops = 2.0 * static_cast<double>(N) * N * N;
    int iterations = 5;

    // reusable benchmarking & verification helper lambda
    auto run_and_verify_kernel = [&](const std::string& kernel_name, dim3 blockSize, dim3 gridSize, auto kernel_invoker) 
    {
        std::cout << "\n----------------------------------------\n";
        std::cout << "Running: " << kernel_name << "\n";
        std::cout << "Block Size: (" << blockSize.x << ", " << blockSize.y << ") | ";
        std::cout << "Grid Size: (" << gridSize.x << ", " << gridSize.y << ")\n";

        // Reset output matrix on device
        CHECK_CUDA(cudaMemset(d_C, 0, bytes));

        // Warm-up run
        kernel_invoker(gridSize, blockSize);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Create CUDA events for precise GPU timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) 
        {
            kernel_invoker(gridSize, blockSize);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        float avg_ms = milliseconds / iterations;
        double gflops = (ops / (avg_ms / 1000.0)) / 1e9;

        std::cout << kernel_name << " Time: " << avg_ms << " ms\n";
        std::cout << kernel_name << " Performance: " << gflops << " GFLOPS\n";

        // --- Result Verification ---
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        bool success = true;
        float expected = static_cast<float>(N);
        for (int i = 0; i < N * N; ++i) 
        {
            if (std::abs(h_C[i] - expected) > 1e-2f) 
            {
                std::cerr << "[ERROR]: Verification FAILED at index " << i 
                          << " (Row: " << i / N << ", Col: " << i % N << "): "
                          << "Expected " << expected << ", Got " << h_C[i] << "\n";
                success = false;
                break;
            }
        }

        if (success) 
        {
            std::cout << "Verification PASSED. All elements equal " << expected << ".\n";
        }
    };



    // ========
    // KERNELS:
    // ========

    // 01 Global Memory Kernel:
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        
        run_and_verify_kernel("Kernel_01 - Global Memory", blockSize, gridSize, [&](dim3 grid, dim3 block) 
        {
            Kernel_01<<<grid, block>>>(d_A, d_B, d_C, N);
        });
    }
    
    // 02 Shared Memory Tiling Kernel:
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + 16 - 1) / 16, (N + 16 - 1) / 16);
        
        run_and_verify_kernel("Kernel_02 - Shared Memory Tiling", blockSize, gridSize, [&](dim3 grid, dim3 block) 
        {
            Kernel_02<<<grid, block>>>(d_A, d_B, d_C, N);
        });
    }

    std::cout << "----------------------------------------\n";

    // Clean up global allocations
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}