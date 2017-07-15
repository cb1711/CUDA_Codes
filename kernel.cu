#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;
__global__ void transpose(int *arr,int *out)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int idy = blockDim.x*threadIdx.x + blockIdx.x;
	out[idy] = arr[idx];
}
__global__ void transposeShared(int* arr, int *out)
{
	__shared__ int share[32][32+2];
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	int idy = blockDim.y*blockIdx.y + threadIdx.y;
	int index = idy*blockDim.x*gridDim.x + idx;
	int tix = threadIdx.x;
	int tiy = threadIdx.y;
	share[tiy][tix] = arr[index];
	__syncthreads();
	int in = tiy*blockDim.x + tix;
	int ir = in / blockDim.y;
	int ic = in % blockDim.y;
	int idyn = blockDim.x*blockIdx.x + ir;
	int idxn = blockDim.y*blockIdx.y + ic;
	int tindex = idyn*blockDim.x*gridDim.x + idxn;

	out[tindex] = share[ic][ir];
}
__global__ void generate(int *arr)
{
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	arr[idx] = idx;
}

int main()
{
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	int *d_arr;
	cudaMalloc((int**)&d_arr, 1024 * 1024 * sizeof(int));
	dim3 blocksize(1024);
	dim3 gridsize(1024);
	dim3 b2(32, 32);
	dim3 g2(32, 32);

	generate <<< gridsize, blocksize >>> (d_arr);
	int *d_out;
	cudaMalloc((int**)&d_out, 1024 * 1024 * sizeof(int));
	//transpose << <gridsize, blocksize >> >(d_arr, d_out);
	transposeShared <<< g2, b2 >>>(d_arr, d_out);
	int *out;
	out = (int*)malloc(sizeof(int) * 1024 * 1024);
	cudaMemcpy(out, d_out, sizeof(int) * 1024 * 1024, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 16; j++)
			cout << out[i*1024+j] << " ";
	cudaDeviceReset();
	return 0;
}