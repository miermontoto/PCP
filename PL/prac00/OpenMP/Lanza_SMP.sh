#!/bin/sh
#
# ----------------------------------------------------------------------
#  Scprip ejemplo para enviar trabajos batch (jobs) OpenMP al gestor de 
#  colas. Los jobs pueden pedir cualquier numero de cores pero se debe
#  tener en cuenta que si demandan mas de los disponibles en el nodo
#  entonces varios procesos/hilos se ejecutaran sobre el mismo core.
#  Para lanzar los trabajos usar:
#     ColaI3   Lanza_SMP.sh // para ejecutar en los Intel I3.  2 cores
#     ColaGPU  Lanza_SMP.sh // para ejecutar en los Intel I5.  4 cores
#     ColaXeon Lanza_SMP.sh // para ejecutar en el Intel Xeon. 8 cores
#  Los nodos Intel I3 siempre estan disponibles. El resto no
# -----------------------------------------------------------------------
#
#
# Para decirle a SGE que sh es el shell del job
#$ -S /bin/sh
#

# Primero Compilamos
make

#Comprobamos que el ejecutable existe
if [ ! -x Hilos ]; then
   echo "Upps, El ejecutable no existe"
   exit 1
fi

#Lanzando la ejecucion
./Hilos 12
