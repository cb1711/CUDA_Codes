#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "gputime.h"
#include <cuda.h>


__global__ void warmup()
{
	//warmup kernel to launch gpu kernels quickly afterwards
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int x = id * 2;
	
}
__device__ unsigned int shared_reduce(unsigned int p, volatile unsigned int * s) {
	int t = threadIdx.x;
	int diff = 16;
	s[t] = p;
	__syncthreads();
	while (diff>0)
	{
		if (t<diff)
		{
			s[t] += s[t + diff];
		}
		diff /= 2;
		__syncthreads();
	}
	return s[0];
}
//Kernel using cuda intrinsic functions __ballot() and __popc()
__device__ unsigned int warp_reduce(unsigned int p, volatile unsigned int * s) {
	//int t = threadIdx.x;
	int c = __ballot(p);//returns a 32 bit integer having ith bit set if the predicate served to the ith thread is true
	int x = __popc(c);//counts the number of set bits in the given 32 bit number
	return x;

}
__global__ void reduce(unsigned int * d_out_shared,
	const unsigned int * d_in)
{
	extern __shared__ unsigned int s[];
	int t = threadIdx.x;
	int p = d_in[t];
	// Intrinsic function use
	unsigned int sr = warp_reduce(p, s);

	//Defined using parallel reduce method
	//unsigned int sr=shared_reduce(p,s);
	if (t == 0)
	{
		*d_out_shared = sr;
	}
}

int main()
{
	const int AR_SIZE = 32;
	const int ARRAY_BYTES = AR_SIZE * sizeof(unsigned int);

	// generate the input array on the host
	unsigned int h_in[AR_SIZE];
	unsigned int sum = 0;
	for (int i = 0; i < AR_SIZE; i++) {
		// generate random float in [0, 1]
		h_in[i] = (float)rand() / (float)RAND_MAX > 0.5f ? 1 : 0;
		sum += h_in[i];
	}

	// declare GPU memory pointers
	unsigned int * d_in, *d_out_shared;

	// allocate GPU memory
	cudaMalloc((void **)&d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_out_shared, sizeof(unsigned int));

	// transfer the input array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	warmup<<<1,1024>>>();
	GpuTimer time;
	time.Start();
	// launch the kernel
	reduce << <1, AR_SIZE, AR_SIZE * sizeof(unsigned int) >> >
		(d_out_shared, d_in);
	time.Stop();

	printf("Your code executed in %g ms\n", time.Elapsed());

	unsigned int h_out_shared;
	// copy back the sum from GPU
	cudaMemcpy(&h_out_shared, d_out_shared, sizeof(unsigned int),
		cudaMemcpyDeviceToHost);

	// compare your resulst against the sum

	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_out_shared);
	return 0;
}