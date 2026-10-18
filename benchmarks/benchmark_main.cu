#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "kernels.cuh"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
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

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Warm-up run
    matrixMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- Benchmark Naive Kernel ---
    std::cout << "Running Naive CUDA Kernel on RTX 5070...\n";
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    int iterations = 5;
    for (int i = 0; i < iterations; ++i) {
        matrixMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_ms = milliseconds / iterations;

    double ops = 2.0 * static_cast<double>(N) * N * N;
    double gflops = (ops / (avg_ms / 1000.0)) / 1e9;

    std::cout << "----------------------------------------\n";
    std::cout << "Naive Kernel Time: " << avg_ms << " ms\n";
    std::cout << "Naive Kernel Performance: " << gflops << " GFLOPS\n";
    std::cout << "----------------------------------------\n";

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}