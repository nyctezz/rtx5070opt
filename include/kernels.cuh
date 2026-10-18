#pragma once
#include <cuda_runtime.h>

// Phase 1: Naive Global Memory Kernel Declaration
__global__ void matrixMulNaive(const float* A, const float* B, float* C, int N);