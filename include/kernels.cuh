#pragma once
#include <cuda_runtime.h>

// Solution 1: Naive Global Memory Kernel Declaration
__global__ void GMKernel(const float* A, const float* B, float* C, int N);