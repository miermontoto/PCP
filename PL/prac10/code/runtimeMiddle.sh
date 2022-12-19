. ./values.sh

params="${xmin} ${xmax} ${ymin} ${maxiter} own sizes 256 512 1024 2048 4096 8192 methods schedule_runtime onlytimes"

unset OMP_NUM_THREADS

export OMP_SCHEDULE="dynamic, 1"
echo "dynamic, 1"
python Launcher.py $params

export OMP_SCHEDULE="dynamic, 2"
echo "dynamic, 2"
python Launcher.py $params

export OMP_SCHEDULE="dynamic, 4"
echo "dynamic, 4"
python Launcher.py $params

export OMP_SCHEDULE="dynamic, 8"
echo "dynamic, 8"
python Launcher.py $params

export OMP_SCHEDULE="dynamic, 16"
echo "dynamic, 16"
python Launcher.py $params

export OMP_SCHEDULE="dynamic, 32"
echo "dynamic, 32"
python Launcher.py $params


