sh origin.sh
. ./values.sh

python Fractal.py $xmin $xmax $ymin $maxiter mandelProf mandelAlumnx sizes 4096 8192 16384 | column -t -s ';'