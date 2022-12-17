. ./values.sh

params="${xmin} ${xmax} ${ymin} ${maxiter} prof py own sizes 256 512 1024 2048 4096 8192"

# Secuencial
export OMP_NUM_THREADS=1
python Launcher.py $params

# Paralelo
unset OMP_NUM_THREADS
python Launcher.py $params noheader -py


