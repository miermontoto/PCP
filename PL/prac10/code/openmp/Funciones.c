#include "Prototipos.h"
#include <omp.h>
#include <xmmintrin.h>

// --- ALGORITMOS DE TIEMPO DE ESCAPE ---
int mandel_iter_while(double x, double y, int maxiter) {

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

int mandel_iter(double x, double y, int maxiter) {

      double u = 0.0, v = 0.0;
      double u2 = 0.0, v2 = 0.0;

      int k;
      for (k = 1; k < maxiter; k++) {
            v = 2.0 * u * v + y;
            u = u2 - v2 + x;
            u2 = u * u;
            v2 = v * v;

            if (u2 + v2 >= 4.0) {
                  return k;
            }
      }

      return maxiter;
}

int mandel_iter_simd(double x, double y, int maxiter) {

      __m128 zero = _mm_set_ps(0.0, 0.0, 0.0, 0.0);
      __m128 two = _mm_set_ps(2.0, 2.0, 2.0, 2.0);
      __m128 four = _mm_set_ps(4.0, 4.0, 4.0, 4.0);

      __m128 u = zero;
      __m128 v = zero;
      __m128 u2 = zero;
      __m128 v2 = zero;

      int k;
      for (k = 1; k < maxiter; k++) {
            v = _mm_add_ps(_mm_mul_ps(two, _mm_mul_ps(u, v)), _mm_set_ps(y, y, y, y));
            u = _mm_add_ps(_mm_sub_ps(u2, v2), _mm_set_ps(x, x, x, x));
            u2 = _mm_mul_ps(u, u);
            v2 = _mm_mul_ps(v, v);

            __m128 uv_sum = _mm_add_ps(u2, v2);
            if (_mm_movemask_ps(_mm_cmpge_ps(uv_sum, four)) != 0) {
                  return k;
            }
      }

      return maxiter;
}

// --- FUNCIONES MANDEL ---
// Función normal, con paralelización típica
void mandel_normal(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

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

void mandel_schedule_runtime(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel for private(i, j, k, c_r, c_im) shared(A) schedule(runtime)
      for (i = 0; i < xres; i++) {
            c_r = xmin + i * dx;
            for (j = 0; j < yres; j++) {
                  c_im = ymin + j * dy;
                  k = mandel_iter(c_r, c_im, maxiter);
                  A[i + j * xres] = k;
            }
      }
}

void mandel_schedule_dynamic(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel for private(i, j, k, c_r, c_im) shared(A) schedule(dynamic)
      for (i = 0; i < xres; i++) {
            c_r = xmin + i * dx;
            for (j = 0; j < yres; j++) {
                  c_im = ymin + j * dy;
                  k = mandel_iter(c_r, c_im, maxiter);
                  A[i + j * xres] = k;
            }
      }
}

void mandel_schedule_dynamic1(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

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

void mandel_schedule_static(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel for private(i, j, k, c_r, c_im) shared(A) schedule(static)
      for (i = 0; i < xres; i++) {
            c_r = xmin + i * dx;
            for (j = 0; j < yres; j++) {
                  c_im = ymin + j * dy;
                  k = mandel_iter(c_r, c_im, maxiter);
                  A[i + j * xres] = k;
            }
      }
}

void mandel_schedule_guided(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel for private(i, j, k, c_r, c_im) shared(A) schedule(guided)
      for (i = 0; i < xres; i++) {
            c_r = xmin + i * dx;
            for (j = 0; j < yres; j++) {
                  c_im = ymin + j * dy;
                  k = mandel_iter(c_r, c_im, maxiter);
                  A[i + j * xres] = k;
            }
      }
}

void mandel_schedule_dynamic32(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel for private(i, j, k, c_r, c_im) shared(A) schedule(dynamic, 32)
      for (i = 0; i < xres; i++) {
            c_r = xmin + i * dx;
            for (j = 0; j < yres; j++) {
                  c_im = ymin + j * dy;
                  k = mandel_iter(c_r, c_im, maxiter);
                  A[i + j * xres] = k;
            }
      }
}

void mandel_schedule_dynamic256(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel for private(i, j, k, c_r, c_im) shared(A) schedule(dynamic, 256)
      for (i = 0; i < xres; i++) {
            c_r = xmin + i * dx;
            for (j = 0; j < yres; j++) {
                  c_im = ymin + j * dy;
                  k = mandel_iter(c_r, c_im, maxiter);
                  A[i + j * xres] = k;
            }
      }
}

void mandel_schedule_dynamic1024(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel for private(i, j, k, c_r, c_im) shared(A) schedule(dynamic, 1024)
      for (i = 0; i < xres; i++) {
            c_r = xmin + i * dx;
            for (j = 0; j < yres; j++) {
                  c_im = ymin + j * dy;
                  k = mandel_iter(c_r, c_im, maxiter);
                  A[i + j * xres] = k;
            }
      }
}

void mandel_schedule_auto(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

      double dx = (xmax - xmin) / xres;
      double dy = (ymax - ymin) / yres;

      double c_r, c_im;

      int i, j, k;
      #pragma omp parallel for private(i, j, k, c_r, c_im) shared(A) schedule(auto)
      for (i = 0; i < xres; i++) {
            c_r = xmin + i * dx;
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
double promedio_normal(int xres, int yres, double* A) {
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
double promedio_vect(int xres, int yres, double* A) {
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
// Se pierde casi el 50% de rendimiento con respecto a la versión con double,
// seguramente por el casting de double a int en cada iteración.
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
