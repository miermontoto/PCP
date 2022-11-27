sh origin.sh
. ./values.sh

params="${xmin} ${xmax} ${ymin} ${maxiter} mandelProf mandelPy mandelAlumnx binarizar sizes 256 512 1024 2048 4096 8192"

# Secuencial
export OMP_NUM_THREADS=1
python Fractal.py $params

# Paralelo
unset OMP_NUM_THREADS
python Fractal.py $params noheader -mandelPy


