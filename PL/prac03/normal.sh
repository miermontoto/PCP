#!/bin/bash
sh EjecutaPrac03.sh

export OMP_NUM_THREADS=1
python3 PRAC03.py MyDGEMM python

unset OMP_NUM_THREADS
python3 PRAC03.py MyDGEMM
