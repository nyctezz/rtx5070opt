#include "kernels.cuh"

// Naive Global Memory Kernel (The Unoptimized Baseline)
__global__ void Kernel_01(const float* A, const float* B, float* C, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) 
    {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) 
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}