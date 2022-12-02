#!/bin/bash
sh origin.sh
. ./values.sh

python Promedio.py $xmin $xmax $ymin $maxiter atomic reduction critical vectorization sizes 256 512 1024 2048 4096 8192 | column -t -s ';'