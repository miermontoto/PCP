#include "Prototipos.h"

int main(int argc, char *argv[])
{
  int
    n, seed, veces, i, j;
  
  double
    *Host_x = NULL,
    *Host_y = NULL,
    *Host_v = NULL,
    *Host_s = NULL;

  /* CUDA and CUBLAS variables */
  int
    ndev,
    ThPerBlk,
    numBlocks;
    
  double
    *Devi_x = NULL,
    *Devi_y = NULL,
    *Devi_v = NULL;

  cudaEvent_t 
    start, stop;
    
  float
    time;
  
  if (argc != 5) {
     printf("Uso: %s <n> <hilos por bloque> <veces> <seed>\n", argv[0]);
     return -1;
  }

  n        = atoi(argv[1]);
  ThPerBlk = atoi(argv[2]);
  veces    = atoi(argv[3]);
  seed     = atoi(argv[4]);

  cudaError_t ret=cudaGetDeviceCount(&ndev);
  if (ndev == 0||ret!=0)
  {
     printf("Error 1: No hay GPU con capacidades CUDA\n");
     return -1;
  }else
     printf("INFO: Hay %d GPUs con capacidades CUDA, seguimos\n", ndev);  


  CHECKNULL(Host_s=(double*)calloc(n,sizeof(double)));

  /* BEGIN NEW */
  CUDAERR(cudaHostAlloc((void **)&Host_x, n*sizeof(double), cudaHostAllocMapped));
  CUDAERR(cudaHostAlloc((void **)&Host_y, n*sizeof(double), cudaHostAllocMapped));
  CUDAERR(cudaHostAlloc((void **)&Host_v, n*sizeof(double), cudaHostAllocMapped));
  
  CUDAERR(cudaHostGetDevicePointer((void **)&Devi_x, (void *)Host_x, 0));
  CUDAERR(cudaHostGetDevicePointer((void **)&Devi_y, (void *)Host_y, 0));
  CUDAERR(cudaHostGetDevicePointer((void **)&Devi_v, (void *)Host_v, 0));
  /* END NEW */

  Genera(Host_x, n, seed);
  Genera(Host_y, n, seed+11);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Resuelve el problema en la CPU */
  cudaEventRecord(start, 0);
     for (i=1; i<=veces; i++)
        for (j=0; j<n; j++) Host_s[j] = Host_x[j] + Host_y[j]; /* OJO, el resultado es Host_s */
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("El tiempo en la CPU     es %2.7E segundos.\n", time/1000.0);

  /* Resuelve el problema en la GPU */
  numBlocks = (n + ThPerBlk - 1) / ThPerBlk;
  cudaEventRecord(start, 0);
     for (i=1; i<=veces; i++)
        kernel_VecAdd<<<numBlocks, ThPerBlk>>>(Devi_v, Devi_x, Devi_y, n);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  CHECKLASTERR();
  printf("El tiempo en la GPU     es %2.7E segundos.\n", time/1000.0);

  printf("El error es %2.7E.\n", Error(n, Host_s, Host_v));

  /* BEGIN NEW */
  CUDAERR(cudaFreeHost(Host_x));
  CUDAERR(cudaFreeHost(Host_y));
  CUDAERR(cudaFreeHost(Host_v));
  /* END NEW */
  free(Host_s);

  return 0;
}
