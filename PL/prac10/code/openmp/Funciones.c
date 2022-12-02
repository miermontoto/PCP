#include "Prototipos.h"
#include <omp.h>

int mandel_iter(double, double, int);

// Función normal, con paralelización típica
void mandel2(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

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

// Función con paralelización utilizando schedule(dynamic, 1)
// Funciona mejor que schedule(static) y valores más altos de dynamic.
// Documentar por qué, probar con otros valores de dynamic y con schedule(guided).
// https://learn.microsoft.com/es-es/cpp/parallel/openmp/d-using-the-schedule-clause
void mandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel for private(i, j, k, c_r, c_im) shared(A) schedule(dynamic, 1)
      for (i = 0; i < xres; i++) {
            c_r = xmin + i * dx;
            for (j = 0; j < yres; j++) {
                  c_im = ymin + j * dy;
                  k = mandel_iter(c_r, c_im, maxiter);
                  A[i + j * xres] = k;
            }
      }
}

void mandel_tasks(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel
            #pragma omp single
            for (i = 0; i < xres; i++) {
                  c_r = xmin + i * dx;

                  #pragma omp task firstprivate(i) private(j, k, c_im)
                  for (j = 0; j < yres; j++) {
                        c_im = ymin + j * dy;
                        k = mandel_iter(c_r, c_im, maxiter);
                        A[i + j * xres] = k;
                  }
            }
}

// --- PROMEDIOS --- //

// Función con variable local para el calculo de cada hilo
// que se junta al final de las ejecuciones.
// No mejora prácticamente nada el rendimiento con respecto a una reducción normal.
// nowait es una adición extra con respecto a las diapositivas que mejora muy ligeramente el rendimiento.
double promedio_atomic(int xres, int yres, double* A) {
      double partialSum, totalSum = 0.0;
      int i, size = xres * yres;

      #pragma omp parallel private(partialSum)
      {
            partialSum = 0.0;

            #pragma omp for nowait
            for (i = 0; i < size; i++) {
                  partialSum += A[i];
            }

            #pragma omp atomic
            totalSum += partialSum;
      }

      return totalSum / size;
}

// Otra versión de la función promedio_atomic, pero utilizando critical.
// Debería ser menos eficiente que la versión anterior, pero tienen el mismo rendimiento.
double promedio_critical(int xres, int yres, double* A) {
      double partialSum, totalSum = 0.0;
      int i, size = xres * yres;

      #pragma omp parallel private(partialSum)
      {
            partialSum = 0.0;

            #pragma omp for nowait
            for (i = 0; i < size; i++) {
                  partialSum += A[i];
            }

            #pragma omp critical
            totalSum += partialSum;
      }

      return totalSum / size;
}


// Función con reducción estándar de OpenMP.
// Para un tamaño de imagen de 4096, el tiempo de ejecución es ≅1.1*10^-2
double promedio(int xres, int yres, double* A) {
      double sum = 0.0;
      int i, size = xres * yres;

      #pragma omp parallel for reduction(+:sum)
      for (i = 0; i < size; i++) {
            sum += A[i];
      }
      return sum / size;
}

// Función de promedio que utiliza vectorización para eliminar la sobrecarga de atomic y reduction.
// https://coderwall.com/p/gocbhg/openmp-improve-reduction-techniques
// Se consigue mejor rendimiento con master que con single.
double promedio_vectorization(int xres, int yres, double* A) {
      double partialSum, totalSum = 0.0;
      double *vect;
      int hilos, i, size = xres * yres;

      #pragma omp parallel
      #pragma omp master
            hilos = omp_get_num_threads();

      vect = (double*) malloc(hilos * sizeof(double));

      #pragma omp parallel private(partialSum)
      {
            partialSum = 0.0;
            #pragma omp for nowait
            for(i = 0; i < size; ++i) {
                  partialSum += A[i];
            }

            vect[omp_get_thread_num()] = partialSum;
      }

      for (i = 0; i < hilos; ++i)
            totalSum += vect[i];

      return totalSum / size;
}

// Versión reduction utilizando un entero para almacenar la suma.
// Se pierde casi el 50% de rendimiento con respecto a la versión con double.
double promedio_int(int xres, int yres, double* A) {
      int sum = 0;
      int i;
      int size = xres * yres;

      #pragma omp parallel for reduction(+:sum)
      for (i = 0; i < size; i++) {
            sum += A[i];
      }
      return sum / (double) size;
}

// no se puede implementar una función promedio_simd ya que la versión de OpenMP (201107) no soporta la directiva.

// --- BINARIZADOS --- //

void binariza(int xres, int yres, double* A, double med) {

      int i, j;
      #pragma omp parallel for private(i, j) shared(A)
      for (i = 0; i < xres; i++) {
            for (j = 0; j < yres; j++) {
                  A[i + j * xres] = A[i + j * xres] > med ? 255 : 0;
            }
      }
}
