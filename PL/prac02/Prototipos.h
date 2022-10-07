/* ********************************************************************** */
/*                     ESTE FICHERO NO DEBE SER MODIFICADO                */
/* ********************************************************************** */
#ifndef PRAC02_H
#define PRAC02_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <cblas.h>

#endif

double Ctimer(void);

void RellenaVector(double *,  const int, const int);
void RellenaMatriz(double **, const int, const int, const int);

double ErrorVector(double *,  double *,  const int);
double ErrorMatriz(double **, double **, const int, const int);

double **newMatrix(const int, const int);
void    freeMatrix(double **, const int n);

void CopiaMatrizRM(double *, double **, const int m, const int n);
