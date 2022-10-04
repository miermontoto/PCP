#!/bin/sh
#
#$ -S /bin/sh
#
# 1.- Compilar usando fichero MakefileMKL en el nodo destino (de ejecucion)
make -f MakefileMKL

# 2.- Comprobando que la libreria exista (la compilacion fue correcta)
if [ ! -x LibBLAS.so ]; then
   echo "Upps, la libreria LibBLAS.so no existe"
   exit 1
fi

TMP1=/opt/intel/oneapi/mkl/latest/lib/intel64
TMP2=/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin
MKL="$TMP1/libmkl_intel_thread.so $TMP1/libmkl_core.so $TMP1/libmkl_intel_lp64.so $TMP2/libiomp5.so"

# 3. Ejecuta en el nodo destino el programa con dimension el contenido del fichero entrada
echo "Secuencial"
export OMP_NUM_THREADS=1
LD_PRELOAD=$MKL python BLAS.py

echo "Paralelo con 4 cores"
export OMP_NUM_THREADS=4
LD_PRELOAD=$MKL python BLAS.py
