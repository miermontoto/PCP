#!/bin/sh
#
#$ -S /bin/sh
#
# 1.- Compilar usando fichero MakefileOpenBLAS en el nodo destino (de ejecucion)
make -f MakefileOpenBLAS

# 2.- Comprobando que la libreria exista (la compilacion fue correcta)
if [ ! -x LibBLAS.so ]; then
   echo "Upps, la libreria LibBLAS.so no existe"
   exit 1
fi

echo $OpenBlas
# 3. Ejecuta en el nodo destino el programa con dimension el contenido del fichero entrada
echo "Secuencial"
export OMP_NUM_THREADS=1
python BLAS.py
 
echo "Paralelo con 4 cores"
export OMP_NUM_THREADS=4
python BLAS.py
