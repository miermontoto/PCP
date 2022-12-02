export LD_PRELOAD="$CUDADIR/lib64/libcudart.so $CUDADIR/lib64/libcublas.so"
. ./values.sh

python Launcher.py $xmin $xmax $ymin $maxiter mandelProf mandelAlumnx sizes 2048 tpb 32 | column -t -s ';'
