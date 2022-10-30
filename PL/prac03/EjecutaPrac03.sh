#!/bin/sh
#
#$ -S /bin/sh
#
# 1.- Compilar usando fichero MakefileMKL en el nodo destino (de ejecucion)
make

# 2.- Comprobando que las librerias existan (la compilacion fue correcta)
if [ ! -x LIBS/PRACIccO0.so ]; then
   echo "Upps, la libreria PRACIccO0.so no existe"
   exit 1
fi
if [ ! -x LIBS/PRACIccO3.so ]; then
   echo "Upps, la libreria PRACIccO3.so no existe"
   exit 1
fi

l3=$(lscpu | grep "L3 cache" | awk '{print $3}')

# 3.- Ejecutar el ejemplo
python PRAC03.py l3
