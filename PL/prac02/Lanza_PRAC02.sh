#!/bin/sh
#$ -S /bin/sh
#

# Primero Compilamos en el nodo destino (de ejecucion)
make

exe="PRAC02SlowGcc PRAC02SlowIcc PRAC02FastGcc PRAC02FastIcc"
m=(1000  500 1000 5000 1000 5000 10000  5000 10000 16000 10000)
n=(500  1000 1000 1000 5000 5000  5000 10000 10000 10000 16000)
seed=2121
echo

# imprimir información sobre el servidor de ejecución
lscpu
lshw

for j in $exe;
do
  i=0
  for loop_i in ${m[@]};
  do
    echo "Ejecutando $j con ${m[$i]} ${n[$i]} $seed"
    EXE/$j ${m[$i]} ${n[$i]} $seed

    i=$(( $i + 1 ))
  done  
done
