export LD_PRELOAD="$CUDADIR/lib64/libcudart.so $CUDADIR/lib64/libcublas.so"
. ./values.sh

python Launcher.py $xmin $xmax $ymin $maxiter own sizes 256 512 1024 2048 4096 8192 methods all | column -t -s ';'
