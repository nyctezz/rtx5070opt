#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "benchmarks.cuh"
#include "kernels.cuh"


// error reporter macro
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

/* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    01_benchmark - matrix multiplication benchmark:

this benchmark evaluates performance of a given kernel for matrix multiplication task

   =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= */

benchmark_01::benchmark_01(int matrices_size)
{
    n = matrices_size;
    bytes = n * n *  sizeof(float);
}

void benchmark_01::init_matrices_with_ones()
{
    h_A.assign(static_cast<size_t>(n) * n, 1.0f);
    h_B.assign(static_cast<size_t>(n) * n, 1.0f);
    h_C.assign(static_cast<size_t>(n) * n, 0.0f);

    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
}

benchmark_result benchmark_01::run(_kernel_func kernel, dim3 block_size) {
    dim3 gridSize((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

    // Pack kernel arguments into an array of pointers
    void* args[] = { &d_A, &d_B, &d_C, &n };

    // Warm-up launch using cudaLaunchKernel
    CHECK_CUDA(cudaLaunchKernel((const void*)kernel, gridSize, block_size, args, 0, 0));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    constexpr int iterations = 5;
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaLaunchKernel((const void*)kernel, gridSize, block_size, args, 0, 0));
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iterations;

    double ops = 2.0 * n * n * n;
    double gflops = ops / (avg_ms * 1e6);

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    float expected = static_cast<float>(n);
    for (size_t i = 0; i < h_C.size(); ++i) {
        if (std::abs(h_C[i] - expected) > 1e-2f) {
            std::cerr << "Verification FAILED at index " << i << '\n';
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            throw std::runtime_error("Kernel verification failed");
        }
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return {avg_ms, gflops};
}

benchmark_01::~benchmark_01()
{
    if (d_A != nullptr) CHECK_CUDA(cudaFree(d_A));
    if (d_B != nullptr) CHECK_CUDA(cudaFree(d_B));
    if (d_C != nullptr) CHECK_CUDA(cudaFree(d_C));
}