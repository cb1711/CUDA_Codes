CUDA code to find transpose of a matrix on a GPU 
For optimization the kernel uses shared memory to make the reads and writes to global memory coalesced
To compile use `nvcc kernel.cu`
