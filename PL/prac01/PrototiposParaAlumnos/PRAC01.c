#include "Prototipos.h"

double Ctimer(void)
{
  struct timeval tm;

  gettimeofday(&tm, NULL);

  return tm.tv_sec + tm.tv_usec/1.0E6;
}

// transpose the vector containing a matrix
double* transpose(double *A, int m, int n) {
  int i, j;
  double *B = (double *)malloc(m*n*sizeof(double));
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      B[i*n+j] = A[j*m+i];
  return B;
}


void MyDGEMMTransposed(int m, int n, int k, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc) {
  A = transpose(A, m, k);

  int i, j, p;
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      double tmp = 0.0;
      for(p = 0; p < k; p++) {
        tmp += A[p + i * ldb] * B[p + j * ldb];
      }
      C[i + j * ldc] = alpha * tmp + beta * C[i + j * ldc];
    }
  }
}

void MyDGEMMNormal(int m, int n, int k, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc) {
  int i, j, p;
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      double tmp = 0.0;
      for(p = 0; p < k; p++) {
        tmp += A[i + p * lda] * B[p + j * ldb];
      }
      C[i + j * ldc] = alpha * tmp + beta * C[i + j * ldc];
    }
  }
}


double MyDGEMM(int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
  double timeini, timefin;
  
  switch (tipo)
  {
    case Normal:
      timeini=Ctimer();
      MyDGEMMNormal(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      timefin=Ctimer()-timeini;  
      break;
    case TransA:
      timeini=Ctimer();  
      MyDGEMMTransposed(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      timefin=Ctimer()-timeini;
      break;
    default:
      timefin=-10;
  }
  return timefin;
}
