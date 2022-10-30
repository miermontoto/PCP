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
  double* At = (double*) malloc(m*n*sizeof(double));
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      At[i*n+j] = A[j*m+i];
  return At;
}

void voidTransposeEqual(double *A, int m, int n) {
  int i, j;
  double* At = (double*) malloc(m*n*sizeof(double));
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      At[i*n+j] = A[j*m+i];
  A = At;
  free(At);
}

void voidTransposeMemcpy(double *A, int m, int n) {
  int i, j;
  double* At = (double*) malloc(m*n*sizeof(double));
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      At[i*n+j] = A[j*m+i];
  memcpy(A, At, m*n*sizeof(double));
  free(At);
}

void voidTransposeCopy(double *A, int m, int n) {
  int i, j;
  double* At = (double*) malloc(m*n*sizeof(double));
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      At[i*n+j] = A[j*m+i];
  for(i = 0; i < m*n; i++)
    A[i] = At[i];
  free(At);
}

void voidTransposeReserve(double *A, double* At, int m, int n) {
  int i, j;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      At[i*n+j] = A[j*m+i];
}

double MyDGEMM(int tipo, int m, int n, int k, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc) {
  int i, j, p;

  if(tipo == TransA) {
    A = transpose(A, m, k);
  }

  #pragma omp parallel for private(j, p)
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      double tmp = 0.0;
      for(p = 0; p < k; p++) {
        double target = A[i + p * lda];
        if(tipo == TransA) {
          target = A[p + i * ldb];
        }
        tmp += target * B[p + j * ldb];
      }
      C[i + j * ldc] = alpha * tmp + beta * C[i + j * ldc];
    }
  }
  return 0.0;
}

double MyDGEMMT(int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) {
  int i, j, p;

  if(tipo == TransA) {
    A = transpose(A, m, k);
  }

  #pragma omp parallel
    #pragma omp single
    for(i = 0; i < m; i++) {

      #pragma omp task firstprivate(i) private(j, p)
      {
        for(j = 0; j < n; j++) {
          double tmp = 0.0;
          for(p = 0; p < k; p++) {
            double target = A[i + p * ldb];
            if(tipo == TransA) {
              target = A[p + i * ldb];
            }
            tmp += target * B[p + j * ldb];
          }
          C[i + j * ldc] = alpha * tmp + beta * C[i + j * ldc];
        }
      }

  }

  return 0.0;
}

double MyDGEMMB(int tipo, int m, int n, int k, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc, int blk) {
  int i, j, p;

  for(i = 0; i < m*n; i++) {
    C[i] = beta * C[i];
  }

  if(tipo == TransA) {A = transpose(A, m, k);}

  for(i = 0; i < m; i += blk) {
    for(j = 0; j < n; j += blk) {
      for(p = 0; p < k; p += blk) {
        double* target = &A[i + p * lda];
        if(tipo == TransA) {
          target = &A[p + i * ldb];
        }
        MyDGEMM(Normal, blk, blk, blk, alpha, target, lda, &B[p + j * ldb], ldb, 1, &C[i + j * ldc], ldc);
      }
    }
  }

  return 0.0;
}



