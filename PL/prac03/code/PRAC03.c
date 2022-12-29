#include "Prototipos.h"

#define TransB 3 // Tipo especial que se utiliza para reutilizar el primer método cuando se trabaja por bloques y de manera traspuesta.

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

double MyDGEMM(int tipo, int m, int n, int k, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc) {
  int i, j, p;
  double tmp;
  int i_plus_jldc, jldb;
  double* At;

  if(tipo == TransA) {
    At = transpose(A, m, k);
  }

  if(tipo > Normal) {
    int ildb;
    #pragma omp parallel for private(j, p)
    for(i = 0; i < m; i++) {
      ildb = i * ldb;

      for(j = 0; j < n; j++) {
        i_plus_jldc = i + j * ldc;
        jldb = j * ldb;

        tmp = 0.0;
        for(p = 0; p < k; p++) {
          tmp += At[p + ildb] * B[p + jldb];
        }

        C[i_plus_jldc] = alpha * tmp + beta * C[i_plus_jldc];
      }
    }
  } else {
    #pragma omp parallel for private(j, p)
    for(i = 0; i < m; i++) {
      for(j = 0; j < n; j++) {
        i_plus_jldc = i + j * ldc;
        tmp = 0.0;
        for(p = 0; p < k; p++) {
          tmp += A[i + p * lda] * B[p + jldb];
        }
        C[i_plus_jldc] = alpha * tmp + beta * C[i_plus_jldc];
      }
    }
  }

  if(tipo == TransA) {
    free(At);
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
  int targetType = Normal;

  for(i = 0; i < m*n; i++) {
    C[i] = beta * C[i];
  }

  if(tipo == TransA) {
    A = transpose(A, m, k);
    targetType = TransB;
  }

  //omp_set_max_active_levels(1);

  //#pragma omp parallel
    //#pragma omp single
    for(i = 0; i < m; i += blk) {

      //#pragma omp task firstprivate(i) private(j, p)
      for(j = 0; j < n; j += blk) {
        for(p = 0; p < k; p += blk) {
          double* target = &A[i + p * lda];
          if(tipo == TransA) {
            target = &A[p + i * ldb];
          }

          MyDGEMM(targetType, blk, blk, blk, alpha, target, lda, &B[p + j * ldb], ldb, 1, &C[i + j * ldc], ldc);
        }
      }
    }

  return 0.0;
}



