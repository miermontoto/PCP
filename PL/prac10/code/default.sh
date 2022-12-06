export LD_PRELOAD="$CUDADIR/lib64/libcudart.so $CUDADIR/lib64/libcublas.so"
. ./values.sh

make all
python Launcher.py $xmin $xmax $ymin $maxiter prof own sizes 1024 bin tpb 32 | column -t -s ';'
