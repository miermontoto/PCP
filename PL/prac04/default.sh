sh origin.sh
. ./values.sh

python Fractal.py $xmin $xmax $ymin $maxiter mandelProf mandelAlumnx tiempos binarizar sizes 4096 | column -t -s ';'