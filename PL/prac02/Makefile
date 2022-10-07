#Para Compilar con Intel C
compi1=icc
flags1Slow=-w2 -O0 -qopenmp
flags1Fast=-w2 -O3 -qopenmp


#Para Compilar con GCC
compi2=gcc
flags2Slow=-Wall -O0 -fopenmp -Wno-unused-variable
flags2Fast=-Wall -O3 -fopenmp -Wno-unused-variable

all :	cleanall creadirec \
	PRAC02SlowIcc PRAC02FastIcc PRAC02SlowGcc PRAC02FastGcc
	@ echo "Compilado "


cleanall : clean
	@rm -rf	EXE

clean :
	@rm -f *.o core *~

creadirec :
	@mkdir -p EXE
	
PRAC02SlowIcc: PRAC02.c Funciones.c
	$(compi1) -o EXE/PRAC02SlowIcc PRAC02.c Funciones.c $(flags1Slow) -lm -qmkl=parallel

PRAC02FastIcc: PRAC02.c Funciones.c
	$(compi1) -o EXE/PRAC02FastIcc PRAC02.c Funciones.c $(flags1Fast) -lm -qmkl=parallel


PRAC02SlowGcc: PRAC02.c Funciones.c
	$(compi2) -o EXE/PRAC02SlowGcc PRAC02.c Funciones.c $(flags2Slow) -L/opt/openblas/lib -lopenblas -lm

PRAC02FastGcc: PRAC02.c Funciones.c
	$(compi2) -o EXE/PRAC02FastGcc PRAC02.c Funciones.c $(flags2Fast) -L/opt/openblas/lib -lopenblas -lm

