#include "Prototipos.h"

int mandel_iter(double, double, int);

void mandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel for private(i, j, k, c_r, c_im) shared(A)
      for (i = 0; i < xres; i++) {
            c_r = xmin + i * dx;
            for (j = 0; j < yres; j++) {
                  c_im = ymin + j * dy;
                  k = mandel_iter(c_r, c_im, maxiter);
                  A[i + j * xres] = k;
            }
      }
}

int mandel_iter(double x, double y, int maxiter) {

      double u = 0.0, v = 0.0;
      double u2 = 0.0, v2 = 0.0;

      int k = 1;
      while (k < maxiter && u2 + v2 < 4.0) {
         v = 2.0 * u * v + y;
         u = u2 - v2 + x;
         u2 = u * u;
         v2 = v * v;
         k++;
      }

      return k == maxiter ? 0 : k;
}


double promedio(int xres, int yres, double* A) {

      double x = 0.0;

      int i, j;
      #pragma omp parallel for private(i, j) shared(A) reduction(+:x)
      for (i = 0; i < xres; i++) {
            for (j = 0; j < yres; j++) {
                  x += A[i + j * xres];
            }
      }

      return x / (xres * yres);
}

double promedio_improved(int xres, int yres, double* A) {
      double partialSum;
      double totalSum = 0.0;

      #pragma omp parallel private(partialSum)
      {
            partialSum = 0.0;
            int i, j;
            #pragma omp for
            for (i = 0; i < xres; i++) {
                  for (j = 0; j < yres; j++) {
                        partialSum += A[i + j * xres];
                  }
            }
            #pragma omp atomic
            totalSum += partialSum;
      }

      return totalSum / (xres * yres);
}


void binariza(int xres, int yres, double* A, double med) {

      int i, j;
      #pragma omp parallel for private(i, j) shared(A)
      for (i = 0; i < xres; i++) {
            for (j = 0; j < yres; j++) {
                  A[i + j * xres] = A[i + j * xres] > med ? 255 : 0;
            }
      }
}

