#include "Prototipos.h"

double Ctimer(void)
{
  struct timeval tm;

  gettimeofday(&tm, NULL);

  return tm.tv_sec + tm.tv_usec/1.0E6;
}

double MyDGEMM(int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
  double timeini, timefin;

  // Lo que el alumno necesite hacer 
  
  switch (tipo)
  {
    case Normal:
      timeini=Ctimer();  
      // llamada a la funcion del alumno normal. Se simula con un timer (sleep)
      sleep(0.5);
      timefin=Ctimer()-timeini;  
      break;
    case TransA:
      timeini=Ctimer();  
      // llamada a la funcion del alumno que trabaja con la transpuesta. Se simula con un timer (sleep)
      sleep(0.5);
      timefin=Ctimer()-timeini;
      break;
    default:
      timefin=-10;
  }
  return timefin;
}
