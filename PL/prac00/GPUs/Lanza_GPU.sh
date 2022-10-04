#!/bin/sh
#
# ---------------------------------------------------------------------
#  Scprip ejemplo para enviar trabajos batch (jobs) a las TESLAs
#  al gestor de colas. Para lanzar los trabajos usar:
#     ColaGPU  Lanza_GPU.sh
# ---------------------------------------------------------------------
#
# Para decirle a SGE que sh es el shell del job
#$ -S /bin/sh
#

# Primero Compilamos en el nodo destino (de ejecucion)
make

#Comprobamos que el ejecutable existe
if [ ! -x VecAdd ]; then
   echo "Upps, El ejecutable no existe"
   exit 1
fi


# ################## MUY IMPORTANTE ######################
#          NO APLICABLE EN LA ACTUALIDAD - OBVIAR
# Se mantiene para si en el futuro volvemos a tener mas
# de una GPU reutilizar. Ranilla 16-11-2018
# Para evitar incluir en los programas el cudasetdevice()
# para selecionar la GPU, se usa la varibale de entorno: 
#    CUDA_VISIBLE_DEVICES=numero
# Tal como estan instaladas las GPUs:
# para la GeForce GT 710
#    numero=1
#
# para la Tesla K20 
#    numero=0
#
# Y para usar las dos es
#    numero=0,1
#
# Ahora lanzamos la ejecucion en SGE con la K20
#export CUDA_VISIBLE_DEVICES=0
# ##################### FIN OBVIAR #######################


#Lanzamos la ejecucion en SGE del ejecutable
#con n=5000 
./VecAdd 5000
