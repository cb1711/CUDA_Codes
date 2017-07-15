
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <algorithm>
using namespace std;
__global__ void generator(int *d_arr)
{
	int id = threadIdx.x;
	d_arr[2*id] = id % 90;
	d_arr[2 * id+1] = id % 107;
}
__global__ void sorter(int *d_arr,int n)
{
	int id = threadIdx.x;
	__shared__ int arr[1024*2];
	arr[2 * id] = d_arr[2 * id];
	arr[2 * id + 1] = d_arr[2 * id + 1];
	for (int i = 0; i < n; i++)
	{
		if (i % 2 == 0)
		{
			int x;
			if (arr[2 * id] > arr[2 * id + 1])
			{
				x = arr[2 * id];
				arr[2 * id] = arr[2 * id + 1];
				arr[2 * id + 1] = x;
			}
		}
		else
		{
			if (id > 0)
			{
				if (arr[2 * id - 1] > arr[2 * id ])
				{
					int x = arr[2 * id ];
					arr[2 * id] = arr[2 * id -1];
					arr[2 * id -1] = x;

				}
			}
		}
		__syncthreads();
	}
	d_arr[2 * id] = arr[2 * id];
	d_arr[2 * id + 1] = arr[2 * id + 1];

}
bool srt(int a, int b)
{
	return a < b;
}
int main()
{
	int *d_arr;
	cudaMalloc((int**)&d_arr, 2*1024 * sizeof(int));
	generator << <1, 1024 >> >(d_arr);
	int arr[2048];
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(arr, d_arr, 2048 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(start, 0);
	sort(arr, arr + 2048, srt);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	sorter << <1, 1024 >> >(d_arr, 2048);
	

	cudaMemcpy(arr, d_arr,2048 * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 2048; i++)
		cout << arr[i] << " ";
	cudaDeviceReset();
	cout << "\n"<<time;
	return 0;

}

