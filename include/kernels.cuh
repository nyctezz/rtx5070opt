#pragma once
#include <cuda_runtime.h>

// Naive Global Memory Kernel Declaration:
__global__ void kernel_01(const float* A, const float* B, float* C, int N);

__global__ void kernel_02(const float* A, const float* B, float* C, int N);