sh origin.sh
. ./values.sh

params="${xmin} ${xmax} ${ymin} ${maxiter} mandelProf mandelPy mandelAlumnx binarizar debug sizes 1024"

# Secuencial
export OMP_NUM_THREADS=1
python Fractal.py $params

# Paralelo
unset OMP_NUM_THREADS
python Fractal.py $params noheader -mandelPy


