sh origin.sh
. ./values.sh

python Fractal.py $xmin $xmax $ymin $maxiter debug diffs mandelProf mandelPy mandelAlumnx sizes 512 | column -t -s ';'