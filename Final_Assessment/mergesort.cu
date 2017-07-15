#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "gputime.h"
#include <cuda.h>
//Implementation using this function runs faster but the answer doesn't match with the answer in the assignment even thoughh the answers appear same
//This is due to the loss of precision on subtracting floats
__device__ float gmax(float a, float b)
{
	return (a > b)*a + (b > a)*b;
}
__global__ void BitonicMergesort(float *d_out,const float *d_in){
	extern __shared__ float smem[];
	int tid = threadIdx.x;
	smem[tid] = d_in[tid];
	smem[tid+32] = d_in[tid+32];
	__syncthreads();
	for (int stage = 1; stage <= 32; stage*=2)
	{
		
		for (int substage = stage; substage >= 1; substage/=2)
		{
			int dir = (tid/stage) % 2;
			int ind = tid + (tid / substage)*substage;
			//Works better compared to comparing and then swapping as it has less branch divergence
			float mx = gmax(smem[ind], smem[ind + substage]);
			float mn = smem[ind] + smem[ind + substage] - mx;
			if (dir == 0)
			{
				/*float x = smem[ind];
				if (x > smem[ind + substage])
				{
					smem[ind] = smem[ind + substage];
					smem[ind + substage] = x;
				}*/
				smem[ind] = mn;
				smem[ind + substage] = mx;
			}
			else
			{
				/*float x = smem[ind];
				if (x < smem[ind + substage])
				{
					smem[ind] = smem[ind + substage];
					smem[ind + substage] = x;
				}*/
				smem[ind] = mx;
				smem[ind + substage] = mn;
			}
			__syncthreads();
		}

	}
	d_out[tid] = smem[tid];
	d_out[tid + 32] = smem[tid + 32];

}
int main()
{
	const int ARRAY_SIZ = 64;
	const int ARRAY_BYTES = ARRAY_SIZ * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZ];
	float h_sorted[ARRAY_SIZ];
	float h_out[ARRAY_SIZ];
	for (int i = 0; i < ARRAY_SIZ; i++) {
		// generate random float in [0, 1]
		h_in[i] = (float)rand() / (float)RAND_MAX;
	}
	// declare GPU memory pointers
	float * d_in, *d_out;

	// allocate GPU memory
	cudaMalloc((void **)&d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_out, ARRAY_BYTES);

	// transfer the input array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	GpuTimer timer;
	timer.Start();
	BitonicMergesort << <1, ARRAY_SIZ/2, ARRAY_SIZ * sizeof(float) >> >(d_out, d_in);
	timer.Stop();
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	for (int i = 0; i < ARRAY_SIZ; i++)
		cout << h_out[i] << " ";
	cudaError_t err = cudaGetLastError();
	cout << cudaGetErrorString(err) << endl;
	cout << "Time taken=" << timer.Elapsed() << endl;
	return 0;

}