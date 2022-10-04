#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void kernelScal(double *A, double alpha, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    A[i] = A[i]*alpha;
}


extern "C" int ScalGPU(double *x, const double alpha, const int n, const int ThpBlk) 
{
   int    NumGPUs, NumBlk;
   double *device_x=NULL;

   if(cudaSuccess != cudaGetDeviceCount(&NumGPUs))
     return -1;
   
   cudaMalloc((void **)&device_x, sizeof(double)*n);
   cudaMemcpy(device_x, x, sizeof(double)*n, cudaMemcpyHostToDevice);

   NumBlk=(n + ThpBlk - 1) / ThpBlk;

   kernelScal<<<NumBlk, ThpBlk>>>(device_x, alpha, n);

   cudaMemcpy(x, device_x, sizeof(double)*n, cudaMemcpyDeviceToHost);
     
   return 0;
}
