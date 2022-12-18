export LD_PRELOAD="$CUDADIR/lib64/libcudart.so $CUDADIR/lib64/libcublas.so"
. ./values.sh

python Launcher.py $xmin $xmax $ymin $maxiter own sizes 2048 | column -t -s ';'
