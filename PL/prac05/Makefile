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

custom1: kernel6_1.cu Funciones.cu
	$(comp) -o VecAdd kernel6_1.cu Funciones.cu $(flags) $(LIBS)

custom1sh: kernel6_1Sh.cu Funciones.cu
	$(comp) -o VecAdd kernel6_1Sh.cu Funciones.cu $(flags) $(LIBS)

custom2: kernel6_2.cu Funciones.cu
	$(comp) -o VecAdd kernel6_2.cu Funciones.cu $(flags) $(LIBS)

custom2sh: kernel6_2Sh.cu Funciones.cu
	$(comp) -o VecAdd kernel6_2Sh.cu Funciones.cu $(flags) $(LIBS)

custom3: kernel6_3.cu Funciones.cu
	$(comp) -o VecAdd kernel6_3.cu Funciones.cu $(flags) $(LIBS)

custom3sh: kernel6_3Sh.cu Funciones.cu
	$(comp) -o VecAdd kernel6_3Sh.cu Funciones.cu $(flags) $(LIBS)

custom4: kernel6_4.cu Funciones.cu
	$(comp) -o VecAdd kernel6_4.cu Funciones.cu $(flags) $(LIBS)