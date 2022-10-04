cflags=-Xcompiler -O3,-Wall,-fPIC
  libs=-lcublas -lcudart

all: cleanall LibGPU 

LibGPU.o: LibGPU.cu
	nvcc $(cflags) -c LibGPU.cu -o LibGPU.o 

LibGPU: LibGPU.o
	g++ -shared -Wl,-soname,LibGPU.so -o LibGPU.so LibGPU.o -L$(CUDADIR)/lib64 $(libs)

clean:
	rm -f *~ *.o core

cleanall: clean
	rm -f LibGPU.so

