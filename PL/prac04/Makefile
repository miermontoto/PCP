all: cleanall mandelAlumnx

mandelAlumnx: Funciones.c
	gcc -Wall -O0 -fopenmp -fPIC -shared Funciones.c -o mandelAlumnx.so 

clean:
	rm -f *~ *.o core

cleanall: clean
	rm -f mandelAlumnx.so

