#include "Prototipos.h"

int main(int argc, char *argv[])
{
  int
    n, seed, veces, i, j;
  
  double
    *Host_x = NULL,
    *Host_y = NULL,
    *Host_v = NULL,
    *Host_s = NULL,
    tiempo;

  /* CUDA and CUBLAS variables */
  int
    ndev,
    ThPerBlk,
    numBlocks;
    
  double
    *Devi_x = NULL,
    *Devi_y = NULL,
    *Devi_v = NULL;

  /* BEGIN NEW */
  cudaEvent_t 
    start, stop;
    
  float
    time;
  /* END NEW */
  
  if (argc != 5) {
     printf("Uso: %s <n> <hilos por bloque> <veces> <seed>\n", argv[0]);
     return -1;
  }

  n        = atoi(argv[1]);
  ThPerBlk = atoi(argv[2]);
  veces    = atoi(argv[3]);
  seed     = atoi(argv[4]);

  CHECKNULL(Host_x=(double*)malloc(n*sizeof(double)));
  CHECKNULL(Host_y=(double*)malloc(n*sizeof(double)));
  CHECKNULL(Host_v=(double*)calloc(n,sizeof(double)));
  CHECKNULL(Host_s=(double*)calloc(n,sizeof(double)));
  Genera(Host_x, n, seed);
  Genera(Host_y, n, seed+11);
  
  cudaError_t ret=cudaGetDeviceCount(&ndev);
  if (ndev == 0||ret!=0)
  {
     printf("Error 1: No hay GPU con capacidades CUDA\n");
     return -1;
  }else
     printf("INFO: Hay %d GPUs con capacidades CUDA, seguimos\n", ndev);  
  
  CUDAERR(cudaMalloc((void **)&Devi_x, n*sizeof(double)));
  CUDAERR(cudaMalloc((void **)&Devi_y, n*sizeof(double)));
  CUDAERR(cudaMalloc((void **)&Devi_v, n*sizeof(double)));

  CUDAERR(cudaMemcpy(Devi_x, Host_x, n*sizeof(double), cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(Devi_y, Host_y, n*sizeof(double), cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(Devi_v, Host_v, n*sizeof(double), cudaMemcpyHostToDevice));

  /* BEGIN NEW */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  /* END NEW */

  /* Resuelve el problema en la CPU */
  /* BEGIN NEW */
  cudaEventRecord(start, 0);
  /* END NEW */
  tiempo=Ctimer();
     for (i=1; i<=veces; i++)
        for (j=0; j<n; j++) Host_v[j] = Host_x[j] + Host_y[j];
  /* BEGIN NEW */
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  /* END NEW */
  tiempo=Ctimer()-tiempo;
  /* BEGIN NEW */
  cudaEventElapsedTime(&time, start, stop);
  printf("El tiempo en la CPU     es %2.7E y %2.7E segundos.\n", tiempo, time/1000.0);
  /* END NEW */

  
  /* Resuelve el problema en la GPU */
  numBlocks = (n + ThPerBlk - 1) / ThPerBlk;
  /* BEGIN NEW */
  cudaEventRecord(start, 0);
  /* END NEW */
  tiempo=Ctimer();
     for (i=1; i<=veces; i++)
        kernel_VecAdd<<<numBlocks, ThPerBlk>>>(Devi_v, Devi_x, Devi_y, n);
  /* BEGIN NEW */
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  /* END NEW */
  tiempo=Ctimer()-tiempo;
  /* BEGIN NEW */
  cudaEventElapsedTime(&time, start, stop);
  CHECKLASTERR();
  printf("El tiempo en la GPU     es %2.7E y %2.7E segundos.\n", tiempo, time/1000.0);
  /* END NEW */

  CUDAERR(cudaMemcpy(Host_s, Devi_v, n*sizeof(double), cudaMemcpyDeviceToHost));
  printf("El error es %2.7E.\n", Error(n, Host_s, Host_v));

  free(Host_x);
  free(Host_y);
  free(Host_v);
  free(Host_s);
  cudaFree(Devi_x);
  cudaFree(Devi_y);
  cudaFree(Devi_v);

  return 0;
}
