#pragma once
#include <vector>
#include <cstddef>

#include <cuda_runtime.h>

using _kernel_func = void(*)(const float*, const float*, float*, int);



struct benchmark_result
{
    float avg_ms;
    double gflops;
};

class benchmark_01 
{
private:
    int n; // square matrix size n x n'
    size_t bytes; // space in memory for one matrix

    std::vector<float> h_A; //h_ for host (in ram)
    std::vector<float> h_B;
    std::vector<float> h_C;

    float *d_A = nullptr; //d_ for device (in gpu)
    float *d_B = nullptr;
    float *d_C = nullptr;


public:
    benchmark_01(int matrices_size);

    void init_matrices_with_ones();
    //void init_matrices_with_random();

    benchmark_result run(_kernel_func kernel, dim3 block_size);

    ~benchmark_01();
};