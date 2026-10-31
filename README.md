<h1 align="center">rtx5070opt</h1>

This project focuses on optymizing Machine Learning / AI calculations on a Nvidia RTX 5070 gpu using custom written kernels in CUDA C++. Below I've put explainations and performance results for different approaches I took while writing more and more advanced solutions.


## Testing Environment

### Hardware

| Component | Specification |
|:---|:---|
| **Motherboard** | ASUS TUF GAMING B650-E WIFI |
| **CPU** | AMD Ryzen 5 9600X (6C / 12T) |
| **GPU** | NVIDIA GeForce RTX 5070 12GB GDDR7 (Palit) |
| **RAM** | 32GB Patriot Memory DDR5-6000 |

### Software

| Component | Version |
|:---|:---|
| **Operating System** | Windows 11 25H2 (Build 26200.9711) |
| **NVIDIA Graphics Driver** | 610.88 |
| **CUDA Driver API (UMD)** | 13.3 |
| **CUDA Toolkit** | 13.4 |
| **CUDA Compiler (NVCC)** | 13.4.92 |




### Kernel 01 - Global Memory:

Simplest possible solution - when multiplying matrices A and B to get C, where A,B,C are n x n size, n x n threads are assigned to calculate their own element in matrix C. Data from A and B is pulled one by one from VRAM which is slow. Threads are often accessing the same data multiple times which is redundant.

