#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void Producto(const int n, const double *a, const double *b, double *c)
{
  int 
    i, j, p;

    for (j=0; j<n; j++)
      for (i=0; i<n; i++)
        for (p=0; p<n; p++)
          c[j*n+i] += a[p*n+i] * b[j*n+p];
}

int main(int argc, char *argv[]) {
  int 
    n, repite, i;

  double
    *a=NULL, *b=NULL, *c=NULL;
  
  if (argc !=3) {
    printf("Uso: %s n repeticiones\n", argv[0]);
    return 0;
  }
  else {
    n     =atoi(argv[1]);
    repite=atoi(argv[2]);
  }

  a  =   (double*) malloc(n*n*sizeof(double));
  b  =   (double*) malloc(n*n*sizeof(double));
  c  =   (double*) malloc(n*n*sizeof(double));

  for (i=0; i<repite; i++)
    Producto(n, a, b, c);

  printf("--Mi ejecucion termino con exito--\n");
  
  free(a);
  free(b);
  free(c);

  return 0;
}
