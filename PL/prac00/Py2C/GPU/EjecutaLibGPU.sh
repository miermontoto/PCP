#!/bin/sh
#
#$ -S /bin/sh
#
# 1.- Compilar usando fichero Makefile en el nodo destino (de ejecucion)
make

# 2.- Comprobando que el ejecutable existe (la compilacion fue correcta)
if [ ! -x LibGPU.so ]; then
   echo "Upps, la libreria LibGPU.so no existe"
   exit 1
fi

export LD_PRELOAD="$CUDADIR/lib64/libcudart.so $CUDADIR/lib64/libcublas.so"

# 3. Ejecuta en el nodo destino el programa con dimension el contenido del fichero entrada
python PythonToGPU.py
