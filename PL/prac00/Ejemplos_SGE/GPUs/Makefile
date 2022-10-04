all: cleanall VecAdd

VecAdd: VecAdd.cu
	nvcc -O2 -o VecAdd VecAdd.cu
	
clean:
	rm -f *~ *.o core

cleanall: clean
	rm -f VecAdd
