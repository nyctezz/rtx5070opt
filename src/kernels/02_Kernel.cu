#include "kernels.cuh"

// Shared Memory Tiling Kernel
__global__ void kernel_02(const float* A, const float* B, float* C, int N) 
{
    //tile dimension (square)
    const int BLOCK_SIZE = 16;

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE]; //sram memory for matrix A
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE]; //sram memory for matrix B

    // local thread IDs inside the block (0 to 15)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // global matrix coordinates for this specific thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //sum for calculating item in matrix C
    float sum = 0.0f;

    // Loop across all tiles needed to cover the width/height of the matrix
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) 
    {
        // 1. Load element into s_A from global memory A

        // We check if our global row is within bounds, and if the current tile column is within bounds
        if (row < N && (t * BLOCK_SIZE + tx) < N) 
        {
            s_A[ty][tx] = A[row * N + (t * BLOCK_SIZE + tx)]; //move from A to s_A
        } 
        else 
        {
            s_A[ty][tx] = 0.0f; // Padding out-of-bounds with zero
        }

        // 2. Load element into s_B from global memory B
        if (col < N && (t * BLOCK_SIZE + ty) < N) 
        {
            s_B[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col]; //move from B to s_B
        } 
        else 
        {
            s_B[ty][tx] = 0.0f; // Padding out-of-bounds with zero
        }


        // Wait for all threads in the block to finish loading their piece of the tile
        __syncthreads();



        //now we have small portion of A and B loaded into sram, calculate small portion of C that we can:
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads(); //because faster thread might overwrite data that slower thread is still using inside the loop
    }

    if (row < N && col < N) 
    {
        C[row * N + col] = sum;
    }
}