#!/bin/sh
#
#$ -S /bin/sh
#

echo "Secuencial para el operador @"
export OMP_NUM_THREADS=1
python Multiply.py

echo "Paralelo con 4 hilos para el operador @"
export OMP_NUM_THREADS=4
python Multiply.py
