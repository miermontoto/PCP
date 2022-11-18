#include "Prototipos.h"

int main(int argc, char *argv[])
{
  int
    n, seed, veces, i, j, rows, cols;

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

  /* BEGIN NEW */
  rows=cols=sqrt(n);
  n=rows*cols;
  printf("Ajustando para que todo discurra bien n=sqrt(n)*sqrt(n)=%d\n",n);
  /* END NEW */

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

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Resuelve el problema en la CPU */
  cudaEventRecord(start, 0);
     for (i=1; i<=veces; i++)
        for (j=0; j<n; j++) Host_v[j] = Host_x[j] + Host_y[j];
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("El tiempo en la CPU     es %2.7E segundos.\n", time/1000.0);

  /* Resuelve el problema en la GPU 1D */
  numBlocks = (n + ThPerBlk - 1) / ThPerBlk;
  cudaEventRecord(start, 0);
     for (i=1; i<=veces; i++)
        kernel_VecAdd1D<<<numBlocks, ThPerBlk>>>(Devi_v, Devi_x, Devi_y, n);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  CHECKLASTERR();
  printf("El tiempo en la GPU1D   es %2.7E.\n", time/1000.0);

  CUDAERR(cudaMemcpy(Host_s, Devi_v, n*sizeof(double), cudaMemcpyDeviceToHost));
  printf("El error es %2.7E.\n", Error(n, Host_s, Host_v));

  /* BEGIN NEW */
  /* Resuelve el problema en la GPU 2D */
  dim3 TPerBlk(ThPerBlk, ThPerBlk);
  dim3 nBlocks((int)ceil((float)cols / ThPerBlk), (int)ceil((float)rows / ThPerBlk));
  cudaEventRecord(start, 0);
     for (i=1; i<=veces; i++)
        kernel_VecAdd2D<<<nBlocks, TPerBlk>>>(Devi_v, Devi_x, Devi_y, rows, cols);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  CHECKLASTERR();
  printf("El tiempo en la GPU2D   es %2.7E.\n", time/1000.0);

  CUDAERR(cudaMemcpy(Host_s, Devi_v, n*sizeof(double), cudaMemcpyDeviceToHost));
  printf("El error es %2.7E.\n", Error(n, Host_s, Host_v));
  /* END NEW */

  free(Host_x);
  free(Host_y);
  free(Host_v);
  free(Host_s);
  cudaFree(Devi_x);
  cudaFree(Devi_y);
  cudaFree(Devi_v);

  return 0;
}
