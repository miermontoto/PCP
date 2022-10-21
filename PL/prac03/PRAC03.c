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

double MyDGEMM(int useless, int m, int n, int k, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc) {
  A = transpose(A, m, k);

  int i, j, p;
  #pragma omp parallel for private(j, p)
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      double tmp = 0.0;
      for(p = 0; p < k; p++) {
        tmp += A[p + i * ldb] * B[p + j * ldb];
      }
      C[i + j * ldc] = alpha * tmp + beta * C[i + j * ldc];
    }
  }

  return 0.0;
}

double MyDGEMMT(int useless, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) {
  A = transpose(A, m, k);

  int i, j, p;

  #pragma omp parallel
    #pragma omp single
    for(i = 0; i < m; i++) {

      #pragma omp task
      {
        for(j = 0; j < n; j++) {
          double tmp = 0.0;
          for(p = 0; p < k; p++) {
            tmp += A[p + i * ldb] * B[p + j * ldb];
          }
          C[i + j * ldc] = alpha * tmp + beta * C[i + j * ldc];
        }
      }

  }

  return 0.0;
}

double MyDGEMMB(int useless, int m, int n, int k, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc, int blk) {
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

  return 0.0;
}



