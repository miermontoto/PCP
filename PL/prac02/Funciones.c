/* ********************************************************************** */
/*                     ESTE FICHERO NO DEBE SER MODIFICADO                */
/* ********************************************************************** */
#include "Prototipos.h"

double Ctimer(void)
{
  struct timeval tm;

  gettimeofday(&tm, NULL);

  return tm.tv_sec + tm.tv_usec/1.0E6;
}


void RellenaVector(double *x, const int n, const int seed) {
   int i=0;

   srand(seed);

   for (i=0; i<n; i++)
    x[i]=((double)(rand()% 1000 + 1))/1.0E3;
}


void RellenaMatriz(double **x, const int n, const int m, const int seed) {
   int i=0, j=0;

   srand(seed);

   for(i=0; i<n; i++)
     for(j=0; j<m; j++)
       x[i][j]=((double)(rand()% 1000 + 1))/1.0E3;
}


double ErrorVector(double *x, double *y, const int n)
{
   int i;
   
   double tmp=0.0;

   for(i=0; i<n; i++)
      tmp += pow(x[i] - y[i], 2.0);
   return sqrt(tmp)/sqrt(n);
}


double ErrorMatriz(double **x, double **y, const int n, const int m)
{
   int i, j;
   
   double tmp=0.0;

   for(i=0; i<n; i++)
     for(j=0; j<m; j++)
       tmp += pow(x[i][j] - y[i][j], 2.0);
   return sqrt(tmp)/sqrt(n*m);
}


double **newMatrix(const int n, const int m) {
   double **M=NULL;
   int    i;
    
   if ((n<=0) || (m<=0)) { return NULL; }
      
   M=(double **)calloc(n, sizeof(double *));
        
   for(i=0; i<n; i++)
     M[i]=(double *)calloc(m, sizeof(double));
              
   return M;
}


void freeMatrix(double **M, const int n)
{
   int i;

   if(M != NULL) {
     for(i=0; i<n; i++) { free(M[i]); M[i]=NULL; }
     free(M);
     M=NULL;
   }
}


void CopiaMatrizRM(double *AVector, double **Matriz, const int m, const int n)
{
  int i;

  for(i=0;i<m;i++)
    memcpy(&AVector[i*n], &Matriz[i][0], n*sizeof(double));
}

