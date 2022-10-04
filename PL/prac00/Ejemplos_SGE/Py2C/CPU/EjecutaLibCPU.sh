#!/bin/sh
#
#$ -S /bin/sh
#
# 1.- Compilar usando fichero Makefile en el nodo destino (de ejecucion)
make

# 2.- Comprobando que el ejecutable existe (la compilacion fue correcta)
if [ ! -x LibCPU.so ]; then
   echo "Upps, la libreria LibCPU.so no existe"
   exit 1
fi

# 3. Ejecuta en el nodo destino el programa con dimension el contenido del fichero entrada
python PythonToC.py
