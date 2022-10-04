# Importar lo necesario
import ctypes
import numpy  as np 
from   numpy.ctypeslib import ndpointer


# Cargar el driver C, llamado LibCPU.so, que tiene una funcion, de nombres MySuma
lib = ctypes.cdll.LoadLibrary('./LibCPU.so')


# Describir la interfaz de la funcion MySuma, que es "double MySuma1(double *, const double *, const double *, const int)"
# y asignarla a un objeto en Python
# .restype  se refiere al tipo de lo que retorna la funcion. Si es void valor poner None
# .argtypes se refiere al tipo de los argumentos de la funcion
SUMAR = lib.MySuma
SUMAR.restype  = ctypes.c_double
SUMAR.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]


# Tamaño de los vectores
n=4096


# Crear los vectores de "entrada" y de "salida"
a=np.random.rand(n).astype('d')
b=np.random.rand(n).astype('d')
c=np.zeros(n).astype('d')


# Llamar a la funcion MySuma de "c" e imprimiendo lo que retorna
ValorC=SUMAR(c, a, b, c.size)


# Comprobando errores
ValorPython=0.0
for i in range(c.size):
  ValorPython+=c[i]

print('El error es '+ str(ValorPython-ValorC))


# NOTA Informativa
# Extrictamente, MySuma seria
# MySuma(np.ascontiguousarray(c, np.float64), np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64), c.size)
# Pero al ser vectores y haber concordancia de tipos python/C no es necesario. 
# Mas aun, en estas condiciones el flags="C_CONTIGUOUS" de los ndpointer tambien podria quitarse
#
# Al ser c un vector c.size retorna el numero de elementos. Si fuese una matriz c.size retorna el numero total 
# de elementos de la matriz, y c.shape retorna el par (m,n) con el numero de filas / columnas
