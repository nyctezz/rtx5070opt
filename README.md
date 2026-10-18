# rtx5070opt

this project focuses on optymizing matrix multiplication efficiency on a Nvidia RTX 5070 gpu using CUDA C++. Below I've put explainations and performance results for different approaches I took while writing more and more advanced Kernels.

I've researched and summarized all important aspects of this gpu in /rtx5070specs, I'll be refering to those later on.

testing rig:
(todo: put my all pc specs here)

## Kernel 01 - Global Memory:

Simplest possible solution - when multiplying matrices A and B to get C, where A,B,C are n x n size, n x n threads are assigned to calculate their own element in matrix C. Data from A and B is pulled one by one from VRAM which is slow. Threads are often accessing the same data multiple times which is redundant.

