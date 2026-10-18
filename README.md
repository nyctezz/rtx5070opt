# rtx5070opt

this project focuses on optymizing matrix multiplication efficiency on a Nvidia RTX 5070 gpu using CUDA C++. Below I've put explainations and performance results for different approaches I took while writing all Kernels.

I've researched and summarized all important aspects of this gpu in /rtx2070specs, I'll be refering to those.

testing rig:
(todo: put my all pc specs here)

## Kernel 01 - Global Memory:
