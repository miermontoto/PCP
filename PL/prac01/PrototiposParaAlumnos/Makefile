all: cleanall creadir PRACGccO0.so PRACGccO3.so PRACIccO0.so PRACIccO3.so 

creadir: 
	@mkdir LIBS
	

cleanall: clean
	@rm -rf	LIBS

clean:
	@rm -f *.o core *~
	

PRACGccO0.so: PRAC01.c
	gcc -o LIBS/PRACGccO0.so -fPIC -shared -Wall -O0 PRAC01.c 

PRACGccO3.so: PRAC01.c
	gcc -o LIBS/PRACGccO3.so -fPIC -shared -Wall -O3 PRAC01.c 


PRACIccO0.so: PRAC01.c
	icc -o LIBS/PRACIccO0.so -fPIC -shared -w2   -O0 PRAC01.c 

PRACIccO3.so: PRAC01.c
	icc -o LIBS/PRACIccO3.so -fPIC -shared -w2   -O3 PRAC01.c 
