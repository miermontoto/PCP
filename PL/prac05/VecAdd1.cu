#include "Prototipos.h"

/* Calcula v[i] = x[i] + y[i] usando CUDA
   Entrada:
         x vector de dimension n
         y vector de dimension n
         h numero de hilos por bloque
   Salida:
         v vector de dimension n
*/

int main(int argc, char *argv[])
{
  int
    n, seed, veces, i, j;

  double
    *Host_x = NULL,
    *Host_y = NULL,
    *Host_v = NULL,
    *Host_s = NULL,
    time;

  /* CUDA and CUBLAS variables */
  int
    ndev,
    ThPerBlk,
    numBlocks;

  double
    *Devi_x = NULL,
    *Devi_y = NULL,
    *Devi_v = NULL;

  if (argc != 5) {
     printf("Uso: %s <n> <hilos por bloque> <veces> <seed>\n", argv[0]);
     return -1;
  }

  n        = atoi(argv[1]);
  ThPerBlk = atoi(argv[2]);
  veces    = atoi(argv[3]);
  seed     = atoi(argv[4]);

  /* Paso 1º */
  Host_x=(double*)malloc(n*sizeof(double));
  Host_y=(double*)malloc(n*sizeof(double));
  Host_v=(double*)calloc(n,sizeof(double));
  Host_s=(double*)calloc(n,sizeof(double));
  CHECKNULL(Host_x=(double*)malloc(n*sizeof(double))); //
  CHECKNULL(Host_y=(double*)malloc(n*sizeof(double))); //
  CHECKNULL(Host_v=(double*)calloc(n,sizeof(double))); //
  CHECKNULL(Host_s=(double*)calloc(n,sizeof(double))); //
  Genera(Host_x, n, seed);
  Genera(Host_y, n, seed+11);


  /* Paso 2º */
  cudaError_t ret=cudaGetDeviceCount(&ndev);
  if (ndev == 0||ret!=0)
  {
     printf("Error 1: No hay GPU con capacidades CUDA\n");
     return -1;
  } else
     printf("INFO: Hay %d GPUs con capacidades CUDA, seguimos\n", ndev);



  /* Paso 3º */
  cudaMalloc((void **)&Devi_x, n*sizeof(double));
  cudaMalloc((void **)&Devi_y, n*sizeof(double));
  cudaMalloc((void **)&Devi_v, n*sizeof(double));
  CUDAERR(cudaMalloc((void **)&Devi_x, n*sizeof(double)));
  CUDAERR(cudaMalloc((void **)&Devi_y, n*sizeof(double)));
  CUDAERR(cudaMalloc((void **)&Devi_v, n*sizeof(double)));

  /* Paso 4º */
  CUDAERR(cudaMemcpy(Devi_x, Host_x, n*sizeof(double), cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(Devi_y, Host_y, n*sizeof(double), cudaMemcpyHostToDevice));
  CUDAERR(cudaMemcpy(Devi_v, Host_v, n*sizeof(double), cudaMemcpyHostToDevice));
  cudaMemcpy(Devi_x, Host_x, n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(Devi_y, Host_y, n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(Devi_v, Host_v, n*sizeof(double), cudaMemcpyHostToDevice);

  /* Resuelve el problema en la CPU */
  time=Ctimer();
     for (i=1; i<=veces; i++)
        for (j=0; j<n; j++) Host_v[j] = Host_x[j] + Host_y[j];
  time=Ctimer()-time;
  printf("El tiempo en la CPU  es %2.7E segundos.\n", time);

  /* Resuelve el problema en la GPU */
  numBlocks = (n + ThPerBlk - 1) / ThPerBlk;
  time=Ctimer();
     for (i=1; i<=veces; i++)
        kernel_VecAdd<<<numBlocks, ThPerBlk>>>(Devi_v, Devi_x, Devi_y, n);
     cudaDeviceSynchronize();
  time=Ctimer()-time;

  /* Paso 5º */
  //CHECKLASTERR();
  printf("El tiempo del kernel CUDA es %2.7E segundos.\n", time);

  /* Paso 6º */
  CUDAERR(cudaMemcpy(Host_s, Devi_v, n*sizeof(double), cudaMemcpyDeviceToHost));
  cudaMemcpy(Host_s, Devi_v, n*sizeof(double), cudaMemcpyDeviceToHost);
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
