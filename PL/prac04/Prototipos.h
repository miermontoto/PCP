/* ********************************************************************** */
/*                     ESTE FICHERO NO DEBE SER MODIFICADO                */
/* ********************************************************************** */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
 
#ifdef OMP
  #include <omp.h>
#endif


void mandel(double, double, double, double, int, int, int, double *);

double media(int, int, double *);

void binariza(int, int, double *, double);
