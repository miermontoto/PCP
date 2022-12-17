export LD_PRELOAD="$CUDADIR/lib64/libcudart.so $CUDADIR/lib64/libcublas.so"
. ./values.sh

python Launcher.py $xmin $xmax $ymin $maxiter prof py own sizes 1024 methods all tpb 32 | column -t -s ';'
