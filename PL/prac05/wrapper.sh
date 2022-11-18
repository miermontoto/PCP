list=(custom1 custom1sh custom2)

for i in ${list[@]}; do
    echo "Running $i"
    make $i
    ./VecAdd 1000000 32 100 2121
done