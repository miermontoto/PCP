cflags=-Xcompiler -fopenmp,-O3,-Wall,-fPIC
libs=-lcublas -lcudart -lgomp

omp: cleanomp mandelAlumnx
cuda: cleancuda mandelGPU
all: cleanall mandelAlumnx mandelGPU

mandelAlumnx:
	gcc -Wall -O0 -fopenmp -fPIC -shared openmp/Funciones.c -o openmp/mandelAlumnx.so

clean:
	rm -f *~ *.o core *.bmp

cleanomp: clean
	rm -f openmp/mandelAlumnx.so

cleancuda: clean
	rm -f cuda/mandelAlumnxGPU.so cuda/mandelGPU.o

cleanall: clean cleanomp cleancuda

mandelGPU.o: cuda/mandelGPU.cu
	nvcc $(cflags) -c cuda/mandelGPU.cu -o cuda/mandelGPU.o

mandelGPU: mandelGPU.o
	g++ -shared -Wl,-soname,mandelGPU.so -o cuda/mandelAlumnxGPU.so cuda/mandelGPU.o -L$(CUDADIR)/lib64 $(libs)
	rm -f cuda/mandelGPU.o
