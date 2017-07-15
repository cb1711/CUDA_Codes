#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "gputime.h"
#include <cuda.h>

#define N 256
#define ARRAY_SIZE 4096

using namespace std;
//Naive smooth kernel without shared memory
__global__ void smooth_naive(float * v, float * v_new) {
	int myIdx = threadIdx.x * gridDim.x + blockIdx.x;
	int numThreads = blockDim.x * gridDim.x;
	int myLeftIdx = (myIdx == 0) ? 0 : myIdx - 1;
	int myRightIdx = (myIdx == (numThreads - 1)) ? numThreads - 1 : myIdx + 1;
	float myElt = v[myIdx];
	float myLeftElt = v[myLeftIdx];
	float myRightElt = v[myRightIdx];
	v_new[myIdx] = 0.25f * myLeftElt + 0.5f * myElt + 0.25f * myRightElt;
}
//Faster kernel using shared memory 
//upto 2.4x faster on GEFORCE 920m
__global__ void smooth(float *arr,float *out)
{
	extern __shared__ float smem[];
	int tid = threadIdx.x;
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	int block = blockIdx.x;
	int mb = gridDim.x;
	smem[tid + 1] = arr[gid];
	if (block == 0)
	{
		if (tid == 0)
		{
			smem[0] = smem[1];
		}
	}
	else
	{
		if (tid == 0)
			smem[0] = arr[gid - 1];
	}
	if (block == mb-1){
		if (tid == N - 1)
		{
			smem[tid + 2] = arr[tid + 1];
		}
		else
		{
			smem[tid + 2] = arr[gid + 1];
		}
	}
	__syncthreads();
	out[gid] = smem[tid] * 0.25f + smem[tid+1] * 0.5f + smem[tid + 2] * 0.25f;
}

__global__ void warmup()
{
	//warmup kernel to launch gpu kernels quickly afterwards
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int x = id * 2;
	
}
int main()
{
	float *d_in, *d_out;
	float h_in[4096],h_out[4096];
	for (int i = 0; i < 4096; i++) {
		h_in[i] = (float)rand() / (float)RAND_MAX;;
	}
	cudaMalloc((float**)&d_in, 4096 * sizeof(float));
	cudaMemcpy(d_in, h_in, sizeof(float)*ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMalloc((float**)&d_out, 4096 * sizeof(float));
	warmup << <1, 1024 >> >();
	GpuTimer timer,timer2;
	timer.Start();
	smooth_naive << < 16, 256 >> >(d_in, d_out);
	timer.Stop();
	cout << timer.Elapsed() << endl;
	//cudaError_t err = cudaGetLastError();
	//cout << cudaGetErrorString(err) << endl;
	timer2.Start();

	smooth <<< 16, 256, 258 * sizeof(float) >> >(d_in, d_out);
	timer2.Stop();
	cout << timer2.Elapsed() << endl;
	cudaMemcpy(h_out, d_out, sizeof(float)*ARRAY_SIZE, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 4096; i++)
	//	cout << h_out[i] << " ";

	cudaFree(d_in);
	cudaFree(d_out);
	cudaDeviceReset();
	
	return 0;
}
