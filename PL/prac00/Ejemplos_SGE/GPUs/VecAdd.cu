/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
        C[i] = A[i] + B[i];
}


int main(int argc, char *argv[])
{
   int numElements=0, i, threadsPerBlock, blocksPerGrid;

   size_t size=0;

   float *h_A=NULL, *h_B=NULL, *h_C=NULL, *d_A=NULL, *d_B=NULL, *d_C=NULL;
   
   if (argc !=2) {
     printf("Uso: %s <dimension vector>\n", argv[0]);
     return 0;
   }

   numElements = atoi(argv[1]);
   size        = numElements * sizeof(float);

   // Allocate the host input vector A
   h_A = (float *)malloc(size);

   // Allocate the host input vector B
   h_B = (float *)malloc(size);

   // Allocate the host output vector C
   h_C = (float *)malloc(size);

   // Verify that allocations succeeded
   if (h_A == NULL || h_B == NULL || h_C == NULL)
   {
      printf("Error reservando memoria para los vectores en el Host!\n");
      return 0;
   }

   // Initialize the host input vectors
   for (i = 0; i < numElements; ++i)
   {
      h_A[i] = rand()/(float)RAND_MAX;
      h_B[i] = rand()/(float)RAND_MAX;
   }

   // Allocate the device input vector A
   if (cudaMalloc((void **)&d_A, size) != cudaSuccess)
   {
      printf("Error reservando memoria para el vector A en la GPU\n");
      return 0;
   }

   // Allocate the device input vector B
   if (cudaMalloc((void **)&d_B, size) != cudaSuccess)
   {
      printf("Error reservando memoria para el vector B en la GPU\n");
      return 0;
   }

   // Allocate the device output vector C
   if (cudaMalloc((void **)&d_C, size) != cudaSuccess)
   {
      printf("Error reservando memoria para el vector C en la GPU\n");
      return 0;
   }

   // Copy the host input vectors A and B in host memory to the device input vectors in
   // device memory
   printf("Copia datos de entrada de la memoria del host a la de la GPU\n");
   if (cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess)
   {
      printf("Fallo copiando datos de entrada de la CPU a la GPU\n");
      return 0;
   }

   if (cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess)
   {
      printf("Fallo copiando datos de entrada de la CPU a la GPU\n");
      return 0;
   }

   // Launch the Vector Add CUDA Kernel
   threadsPerBlock = 256;
   blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
   printf("CUDA kernel launched con %d blocks de %d threads\n", blocksPerGrid, threadsPerBlock);

   vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

   if (cudaGetLastError() != cudaSuccess)
   {
      printf("Fallo en la ejecucion del kernel !!!\n");
      return 0;
   }

   // Copy the device result vector in device memory to the host result vector
   // in host memory.
   printf("Copy output data from the CUDA device to the host memory\n");
   if (cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) != cudaSuccess)
   {
      printf("Fallo copiando vector C de la GPU a la CPU\n");
      return 0;
   }

   // Verify that the result vector is correct
   for (i = 0; i < numElements; ++i)
   {
      if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
      {
         printf("Upps, se ha sumado mal en la posicion %d\n", i);
         return 0;
      }
   }

   // Free device global memory
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   // Free host memory
   free(h_A);
   free(h_B);
   free(h_C);

   printf("DONE, todo OK\n");

   return 0;
}

