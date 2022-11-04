import sys
from PIL import Image
import ctypes
import numpy as np
from numpy import linalg as LA
from numpy.ctypeslib import ndpointer
from time import time



#########################################################################
# Prepara gestión librería externa de Profesor 	(NO MODIFICAR)		#
#########################################################################
libProf = ctypes.cdll.LoadLibrary('./mandelProf.so')
# Preparando para el uso de "void mandel(double, double, double, double, int, int, int, double *)"
# .restype  se refiere al tipo de lo que retorna la funcion. Si es void, valor "None".
# .argtypes se refiere al tipo de los argumentos de la funcion.
# Hay tres funciones en la librería compartida: mandel(...), promedio(...) y binariza(...).
# Todas ellas funcionan en paralelo si hay más de un hilo disponible.
mandelProf = libProf.mandel

mandelProf.restype  = None
mandelProf.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Preparando para el uso de "double promedio(int, int, double *)"
mediaProf = libProf.promedio

mediaProf.restype  = ctypes.c_double
mediaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Preparando para el uso de "void binariza(int, int, double *, double)"
binarizaProf = libProf.binariza

binarizaProf.restype  = None
binarizaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double]


#########################################################################
# Preparar gestión librería externa de Alumnx llamada mandelAlumnx.so	#
#########################################################################




#########################################################################
# Función en Python para resolver el cálculo del fractal	        	#
# Codificar el algoritmo que, para los parámetros dados, calcula    	#
# el fractal siguiendo el pseudocódigo del guion. Resultado en vector A	#
#########################################################################
def mandelPy(xmin, ymin, xmax, ymax, maxiter, xres, yres, A):
    return


#########################################################################
# 	Función para guardar la imagen a archivo (NO MODIFICAR)		        #
#########################################################################
def grabar(vect, xres, yres, output):
    A2D=vect.astype(np.ubyte).reshape(yres,xres) #row-major por defecto
    im=Image.fromarray(A2D)
    im.save(outputfile)
    print(f"Grabada imagen como {outputfile}")


#########################################################################
# 		                        	MAIN					           	#
#########################################################################
if __name__ == "__main__":
    #  Procesado de los agumentos					(NO MODIFICAR)	#
    if len(sys.argv) != 7:
        print('\033[91m'+'USO: main.py <xmin> <xmax> <ymin> <yres> <maxiter> <outputfile>')
        print("Ejemplo: -0.7489 -0.74925 0.1 1024 1000 out.bmp"+'\033[0m')
        sys.exit(2)

    xmin=float(sys.argv[1])
    xmax=float(sys.argv[2])
    ymin=float(sys.argv[3])
    yres=int(sys.argv[4])
    maxiter=int(sys.argv[5])
    outputfile = sys.argv[6]

    #  Cálculo de otras variables necesarias						#
    xres=...
    ymax=...



    #  Reserva de memoria de las imágenes en 1D	(AÑADIR TANTAS COMO SEAN NECESARIAS)	#
    fractalPy = np.zeros(yres*xres).astype(np.double) #Esta es para el alumnado, versión python
    fractalProf = np.zeros(yres*xres).astype(np.double) #Esta es para el profesor





    #  Comienzan las ejecuciones							#
    print(f'\nCalculando fractal de {yres}x{xres} maxiter:{maxiter}:')

    #  Llamada a la función de cálculo del fractal en python (versión alumnx)	(NO MODIFICAR) #
    sPy = time()
    mandelPy(xmin, ymin, xmax, ymax, maxiter, xres, yres, fractalPy)
    sPy = time()- sPy
    print(f"mandelPy		ha tardado {sPy:1.5E} segundos")

    #  Llamada a la función de cálculo del fractal en C (versión profesor) (NO MODIFICAR) #
    sC = time()
    mandelProf(xmin, ymin, xmax, ymax, maxiter, xres, yres, fractalProf)
    sC = time()- sC
    print(f"mandelC (prof)		ha tardado {sC:1.5E} segundos")

    #  Comprobación del error de cálculo del fractal en python (versión alumnx frente a prof) (No MODIFICAR)#
    print('El error es '+ str(LA.norm(fractalPy-fractalProf)))


    #  Llamada a la función de cálculo del fractal en C (versión alumnx). 		#




    #  Comprobación del error de cálculo del fractal en C (versión alumnx frente a prof)#



    #  Llamada a la función de cálculo de la media (versión profesor) 	(NO MODIFICAR) 	#
    sM = time()
    promedioProf=mediaProf(xres, yres, fractalProf)
    sM = time()- sM
    print(f"Promedio (prof)={promedioProf:1.3E}	ha tardado {sM:1.5E} segundos")

    #  Llamada a la función de cálculo de la media en C (versión alumnx)		#




    #  Comprobación del error en el promedio en C (versión alumnx frente a prof)	#




    #  Llamada a la función de cálculo del binarizado (versión profesor) (NO MODIFICAR)	#
    sB = time()
    binarizaProf(xres, yres, fractalProf, promedioProf)
    sB = time()- sB
    print(f"Binariza (prof)		ha tardado {sB:1.5E} segundos")

    #  Llamada a la función de cálculo del binarizado en C (versión alumnx)		#




    #  Comprobación del error en el binarizado en C (versión alumnx) 			#




    #  Grabar a archivo	la imagen que se desee (SOLO PARA DEPURAR)			#
    grabar(fractalPy,xres,yres,outputfile)



