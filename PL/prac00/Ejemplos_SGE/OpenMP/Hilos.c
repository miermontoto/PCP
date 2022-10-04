#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

int main(int argc, char *argv[])
{
 int NumeroCores, NumeroHilos, NumeroUsado;

 double Tiempo_OpenMP;
   
 Tiempo_OpenMP=omp_get_wtime();

 NumeroCores=omp_get_num_procs();

 if (argc == 2)
   NumeroHilos=atoi(argv[1]);
 else
   NumeroHilos=2;

 NumeroUsado=NumeroHilos-1;
 omp_set_num_threads(NumeroUsado);

 printf("Informacion:\n");
 printf("  Numero total de Cores: %d\n", NumeroCores);
 printf("  Numero de Cores disponibles %d\n", NumeroHilos);
 printf("  Numero de hilos que usamos  %d\n", NumeroUsado);

 Tiempo_OpenMP=omp_get_wtime();

 #pragma omp parallel
 {
   int Hilo_ID;

   Hilo_ID = omp_get_thread_num();
   printf("Soy el hilo %d\n",Hilo_ID);
   system("uname -n");
   system("sleep 10");
 }

 Tiempo_OpenMP=omp_get_wtime()-Tiempo_OpenMP;
 printf("\nConsumidos %lf segundos en este ejemplo\n", Tiempo_OpenMP);

 return 0;
}
