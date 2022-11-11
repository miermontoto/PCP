/* ********************************************************************** */
/*                     ESTE FICHERO NO DEBE SER MODIFICADO                */
/* ********************************************************************** */
#ifndef PRAC01_H
#define PRAC01_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

#define Normal 1
#define TransA 2

#endif

double Ctimer(void);

double MyDGEMM(int, int, int, int, double, double*, int, double*, int, double, double*, int);
