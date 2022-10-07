#include "Prototipos.h"

/* Resuelve, en secuencial y paralelo, el producto matriz*vector x=A*v
   Tipo Almacenamiento: 2D (estandar en C) y 1D row-major (compatible 2D C)
   Entrada:
         A matriz de dimensiones m x n
         v vector de dimensiones n x 1
   Intermedio:
         AVector matriz A (m x n) almacenada como vector usando 1D row-major
   Salida:
         x vector de dimensiones m x 1 
*/

int main(int argc, char *argv[])
{
  int
    m, n, seed, i, j, k;
  
  double
    **Matriz=NULL,
    *MVector=NULL,
    *VectorV=NULL,
    *VectXse=NULL,
    *VectXpa=NULL,
    *VectXcb=NULL,
        
    sumas=0, sumap=0,    
    timeini, timefin;

  if (argc != 4) {
     printf("Uso: %s <m> <n> <seed>\n", argv[0]);
     return -1;
  }

  m   =atoi(argv[1]);
  n   =atoi(argv[2]);
  seed=atoi(argv[3]);


  /* Comprobando que ninguna dimension es nula */
  if ((m==0) || (n==0)) {
     printf("Error: Alguna dimension es nula.\n");
     return -1;
  }


  /* Reservando memoria */
  Matriz =newMatrix(m, n);
  MVector=(double *)calloc(m*n, sizeof(double));
  VectorV=(double *)calloc(n,   sizeof(double));
  VectXse=(double *)calloc(m,   sizeof(double));
  VectXpa=(double *)calloc(m,   sizeof(double));
  VectXcb=(double *)calloc(m,   sizeof(double));


  /* Rellenando con datos la matriz y el vector de entrada */
  RellenaMatriz(Matriz,  m, n, seed);
  RellenaVector(VectorV, n,    seed+11);
  CopiaMatrizRM(MVector, Matriz, m, n);


  /* ******************************************************************************* */
  /*                                                                                 */
  /*                                    FASE 0                                       */
  /*                                                                                 */
  /* ******************************************************************************* */
  timeini=Ctimer();
     /* **************************************************************************** */
     /*                                                                              */
     /* RESOLVIENDO EL PROBLEMA USANDO LIBRERIAS DE ALTO RENDIMIENTO. ATLAS PARA GCC */
     /* Y MKL PARA ICC                                                               */
     /*                                                                              */
     /* **************************************************************************** */
     cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, MVector, n, VectorV, 1, 0.0, VectXcb, 1);
  timefin=Ctimer();  
  printf("\n\nTiempo FASE 0: Con librerias de alto rendimiento   %2.7E sec.\n", timefin-timeini);


  /* ******************************************************************************* */
  /*                                                                                 */
  /*                                    FASE 1                                       */
  /*                                                                                 */
  /* ******************************************************************************* */
  timeini=Ctimer();
     /* **************************************************************************** */
     /*                                                                              */
     /* EL ALUMNO DEBE INCORPORAR AQUI EL CODIGO PARA REALIZA EL PRODUCTO MATRIZ POR */
     /* VECTOR DE FORMA SECUENCIAL. AUNQUE SEA POCO EFICIENTE NO USE VARIABLES       */
     /* AUXILIARES PARA ALMACENAR PARCIALES, TRABAJAR DIRECTAMENTE SOBRE VectXse      */
     /* EL RESULTADO FINAL SE DEBE DEJAR EN VectXse                                   */
     /* **************************************************************************** */
  timefin=Ctimer();  
  printf("\n\nTiempo FASE 1: Secuencial sin variables auxiliares %2.7E sec. Error %1.5E.\n", timefin-timeini, ErrorVector(VectXse, VectXcb, m));


  /* ******************************************************************************* */
  /*                                                                                 */
  /*                                    FASE 2                                       */
  /*                                                                                 */
  /* ******************************************************************************* */
  timeini=Ctimer();
     /* **************************************************************************** */
     /*                                                                              */
     /* IGUAL QUE FASE 1 PERO USANDO VARIABLES AUXILIARES PARA ALMACENAR RESULTADOS  */
     /* PARCIALES / INTERMEDIOS                                                      */
     /* EL RESULTADO FINAL SE DEBE DEJAR EN VectXse                                   */
     /* **************************************************************************** */
  timefin=Ctimer();  
  printf("Tiempo FASE 2: Secuencial con variables auxiliares %2.7E sec. Error %1.5E.\n", timefin-timeini, ErrorVector(VectXse, VectXcb, m));


  /* ******************************************************************************* */
  /*                                                                                 */
  /*                                    FASE 3                                       */
  /*                                                                                 */
  /* ******************************************************************************* */
  timeini=Ctimer();  
     /* **************************************************************************** */
     /*                                                                              */
     /* IGUAL QUE FASE 2 PERO USANDO MVECTOR (MATRIZ ALMACENADA USANDO 1D ROW-MAJOR) */
     /* EN VEZ DE MATRIZ (MATRIZ ALMACENADA USANDO 2D)                               */
     /* EL RESULTADO FINAL SE DEBE DEJAR EN VectXse                                   */
     /* **************************************************************************** */
  timefin=Ctimer();  
  printf("Tiempo FASE 3: Secuencial Matriz con 1D Row-Major  %2.7E sec. Error %1.5E.\n", timefin-timeini, ErrorVector(VectXse, VectXcb, m));




  /* ******************************************************************************* */
  /*                                                                                 */
  /*                                    FASE 4                                       */
  /*                                                                                 */
  /* ******************************************************************************* */
  timeini=Ctimer();  
     /* **************************************************************************** */
     /*                                                                              */
     /* EL ALUMNO DEBE INCORPORAR AQUI EL CODIGO PARA REALIZA EL PRODUCTO MATRIZ POR */
     /* VECTOR DE FORMA PARALELA. AUNQUE SEA POCO EFICIENTE NO USE VARIABLES         */
     /* AUXILIARES PARA ALMACENAR PARCIALES, TRABAJAR DIRECTAMENTE SOBRE VectXpa      */
     /* EL RESULTADO FINAL SE DEBE DEJAR EN VectXpa                                   */
     /* **************************************************************************** */

     /* ELIMINAR "//" EN LA SIGUENTE LINEA PARA USAR PARALELISMO CUANDO SE NECESITE  */
     //#pragma omp parallel for private(j)
  timefin=Ctimer();
  printf("\n\nTiempo FASE 4: Paralelo sin variables auxiliares %2.7E sec. Error %1.5E.\n", timefin-timeini, ErrorVector(VectXpa, VectXcb, m));


  /* ******************************************************************************* */
  /*                                                                                 */
  /*                                    FASE 5                                       */
  /*                                                                                 */
  /* ******************************************************************************* */
  timeini=Ctimer();  
     /* **************************************************************************** */
     /*                                                                              */
     /* IGUAL QUE FASE 4 PERO USANDO VARIABLES AUXILIARES GLOBALES PARA ALMACENAR    */
     /* RESULTADOS PARCIALES / INTERMEDIOS                                           */
     /* EL RESULTADO FINAL SE DEBE DEJAR EN VectXpa                                   */
     /* **************************************************************************** */

     /* ELIMINAR "//" EN LA SIGUENTE LINEA PARA USAR PARALELISMO CUANDO SE NECESITE  */
     //#pragma omp parallel for private(j, sumap)
  timefin=Ctimer();
  printf("Tiempo FASE 5: Paralelo con auxiliares globales  %2.7E sec. Error %1.5E.\n", timefin-timeini, ErrorVector(VectXpa, VectXcb, m));


  /* ******************************************************************************* */
  /*                                                                                 */
  /*                                    FASE 6                                       */
  /*                                                                                 */
  /* ******************************************************************************* */
  timeini=Ctimer();  
     /* **************************************************************************** */
     /*                                                                              */
     /* IGUAL QUE FASE 4 PERO USANDO VARIABLES AUXILIARES LOCALES PARA ALMACENAR     */
     /* RESULTADOS PARCIALES / INTERMEDIOS                                           */
     /* EL RESULTADO FINAL SE DEBE DEJAR EN VectXpa                                   */
     /* **************************************************************************** */

     /* ELIMINAR "//" EN LA SIGUENTE LINEA PARA USAR PARALELISMO CUANDO SE NECESITE  */
     //#pragma omp parallel for
  timefin=Ctimer();
  printf("Tiempo FASE 6: Paralelo con auxiliares locales   %2.7E sec. Error %1.5E.\n", timefin-timeini, ErrorVector(VectXpa, VectXcb, m));


  /* ******************************************************************************* */
  /*                                                                                 */
  /*                                    FASE 7                                       */
  /*                                                                                 */
  /* ******************************************************************************* */
  timeini=Ctimer();  
     /* **************************************************************************** */
     /*                                                                              */
     /* IGUAL QUE FASE 6 PERO USANDO MVECTOR (MATRIZ ALMACENADA USANDO 1D ROW-MAJOR) */
     /* EN VEZ DE MATRIZ (MATRIZ ALMACENADA USANDO 2D)                               */
     /* EL RESULTADO FINAL SE DEBE DEJAR EN VectXpa                                   */
     /* **************************************************************************** */

     /* ELIMINAR "//" EN LA SIGUENTE LINEA PARA USAR PARALELISMO CUANDO SE NECESITE  */
     //#pragma omp parallel for
  timefin=Ctimer();  
  printf("Tiempo FASE 7: Paralelo Matriz con 1D Row-Major  %2.7E sec. Error %1.5E.\n", timefin-timeini, ErrorVector(VectXpa, VectXcb, m));




  /* ******************************************************************************* */
  /*                                                                                 */
  /*                                    FASE 8                                       */
  /*                                                                                 */
  /* ******************************************************************************* */
  timeini=Ctimer();  
     /* **************************************************************************** */
     /*                                                                              */
     /* AHORA SUME EN SECUENCIAL LOS ELEMENTOS DEL VECTOR RESULTANTE VectXse         */
     /* EL RESULTADO DEBE GUARDARDE EN LA VARIABLE sumas                             */
     /* **************************************************************************** */
  timefin=Ctimer();
  printf("\n\nTiempo FASE 8: Operador de reducción en secuencial %2.7E sec.\n", timefin-timeini);



  /* ******************************************************************************* */
  /*                                                                                 */
  /*                                    FASE 9                                       */
  /*                                                                                 */
  /* ******************************************************************************* */
  timeini=Ctimer();  
     /* **************************************************************************** */
     /*                                                                              */
     /* AHORA SUME EN PARALELO LOS ELEMENTOS DEL VECTOR RESULTANTE VectXpa           */
     /* EL RESULTADO DEBE GUARDARDE EN LA VARIABLE sumap                             */
     /* **************************************************************************** */

     /* ELIMINAR "//" EN LA SIGUENTE LINEA PARA USAR PARALELISMO CUANDO SE NECESITE  */
     //#pragma omp parallel for reduction(+:sumap)
  timefin=Ctimer();
  printf("Tiempo FASE 9: Operador de reducción en paralelo   %2.7E sec. Error %1.5E.\n", timefin-timeini, fabs((sumap-sumas)/(double)m));





  /* Liberando memoria */
  free(VectorV);
  free(VectXse);
  free(VectXpa);
  free(MVector);
  freeMatrix(Matriz, m);

  return 0;
}
