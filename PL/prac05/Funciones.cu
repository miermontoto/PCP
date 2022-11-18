#include "Prototipos.h"

double Ctimer(void)
{
  struct timeval tm;

  gettimeofday(&tm, NULL);

  return tm.tv_sec + tm.tv_usec/1.0E6;
}



/* Rellena por filas los elementos de una matriz A de dimensiones m x n */
void Genera(double *A, int n, int seed) {
   int i=0;

   srand(seed);

   for (i=0; i<n; i++)
     A[i] = ((double)(rand()% 1000 + 1))/1.0E3;
}

/* Calculando el error con norma Frobenius */
double Error(int n, double *X, double *y)
{
   int i;

   double tmp, error=0.0;

   for (i=0; i<n; i++)
   {
      tmp = X[i] - y[i];
      error += tmp*tmp;
   }

   return sqrt(error/n);
}


__global__ void kernel_VecAdd(double *v, const double *x, const double *y, const int size)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   if (tid < size)
      v[tid] = x[tid] + y[tid];
}


__global__ void kernel_VecAdd1D(double *v, const double *x, const double *y, const int size)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   if (tid < size)
      v[tid] = x[tid] + y[tid];
}


__global__ void kernel_VecAdd2D(double *v, const double *x, const double *y, const int rows, const int cols)
{
   int X = blockIdx.x * blockDim.x + threadIdx.x;
   int Y = blockIdx.y * blockDim.y + threadIdx.y;

   if (Y<rows && X<cols)
      v[Y*cols + X] = x[Y*cols + X] + y[Y*cols + X];
}

__global__ void kernel6_1(double *x, double *y, const int size) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   if (tid < size) {
      x[tid] = y[tid] * y[tid] + x[tid];
   }
}

__global__ void kernel6_2(double *x, double *y, double *A, const int n) {

   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   if (tid < n) {
      y[tid] = 0.0;
      for (int i=0; i<n; i++)
         y[tid] += A[tid*n + i] * x[i];
   }
}
