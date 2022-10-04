#include <stdio.h>
#include <stdlib.h>

double MySuma(double *c, const double *a, const double *b, const int n) 
{
   double sum=0;
   int    i;
   
   for(i=0; i<n; i++)
   {
     c[i] = a[i]+b[i];
     sum += c[i];
   }
   return sum;
}
