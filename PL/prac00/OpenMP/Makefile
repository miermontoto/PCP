all: cleanall Hilos

Hilos: Hilos.c
	gcc -O3 -fopenmp -o Hilos Hilos.c

clean:
	rm -f *~ *.o core

cleanall: clean
	rm -f Hilos
