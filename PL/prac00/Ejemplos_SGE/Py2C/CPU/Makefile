all: cleanall LibCPU

LibCPU: LibCPU.c
	gcc -Wall -O3 -fPIC -shared LibCPU.c -o LibCPU.so

clean:
	rm -f *~ *.o core

cleanall: clean
	rm -f LibCPU.so

