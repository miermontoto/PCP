all: cleanall creadir PRACIccO0.so PRACIccO3.so 


creadir: 
	@mkdir LIBS
	
cleanall: clean
	@rm -rf	LIBS

clean:
	@rm -f *.o core *~
	

PRACIccO0.so: PRAC03.c
	icc -o LIBS/PRACIccO0.so -qopenmp -fPIC -shared -w2 -O0 PRAC03.c

PRACIccO3.so: PRAC03.c
	icc -o LIBS/PRACIccO3.so -qopenmp -fPIC -shared -w2 -O3 PRAC03.c
