#include "kernels.cuh"

// Naive Global Memory Kernel (The Unoptimized Baseline)
__global__ void kernel_01(const float* A, const float* B, float* C, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) 
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) 
        {
            sum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}