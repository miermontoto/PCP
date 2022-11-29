#Para Compilar con nvcc
comp=nvcc
flags=-Xcompiler -O3,-Wall
LIBS=-lcurand -lcudart

all: VecAdd1 VecAdd2 VecAdd3 VecAdd4 VecAdd5
	@ echo "Compilado "

cleanall : clean
	@rm -f VecAdd

clean :
	@rm -f *.o core *~


VecAdd1: VecAdd1.cu Funciones.cu
	$(comp) -o VecAdd VecAdd1.cu Funciones.cu $(flags) $(LIBS)

VecAdd2: VecAdd2.cu Funciones.cu
	$(comp) -o VecAdd VecAdd2.cu Funciones.cu $(flags) $(LIBS)

VecAdd3: VecAdd3.cu Funciones.cu
	$(comp) -o VecAdd VecAdd3.cu Funciones.cu $(flags) $(LIBS)

VecAdd4: VecAdd4.cu Funciones.cu
	$(comp) -o VecAdd VecAdd4.cu Funciones.cu $(flags) $(LIBS)

VecAdd5: VecAdd5.cu Funciones.cu
	$(comp) -o VecAdd VecAdd5.cu Funciones.cu $(flags) $(LIBS)

custom: kernel6_base.cu Funciones.cu
	$(comp) -o VecAdd kernel6_base.cu Funciones.cu $(flags) $(LIBS)