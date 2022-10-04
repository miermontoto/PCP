#!/bin/sh
#
# ---------------------------------------------------------------------
#  Scprip ejemplo para enviar trabajos batch (jobs) secuenciales
#  al gestor de colas. Los jobs secuenciales SOLO demandan (usan)
#  un core. Para lanzar los trabajos usar:
#     ColaI3   Lanza_Secuencial.sh  // para ejecutar en los Intel I3
#     ColaGPU  Lanza_Secuencial.sh  // para ejecutar en los Intel I5
#     ColaXeon Lanza_Secuencial.sh  // para ejecutar en el  Intel Xeon
#  Los nodos Intel I3 siempre estan disponibles. El resto no
# ---------------------------------------------------------------------
#
# Para decirle a SGE que sh es el shell del job
#$ -S /bin/sh
#

# Primero Compilamos en el nodo destino (de ejecucion)
make

#Comprobamos que el ejecutable existe
if [ ! -x Naive ]; then
   echo "Upps, El ejecutable no existe"
   exit 1
fi

#Lanzamos la ejecucion en SGE del ejecutable
#con n=100 y 3 repeticiones
./Naive 100 3
