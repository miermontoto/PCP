# Importar lo necesario
import ctypes
import numpy  as np 
from   numpy.ctypeslib import ndpointer
from   numpy           import linalg as LA


# Cargar el driver CUDA, llamado LibGPU.so, que tiene una funcion, de nombre ScalGPU
lib = ctypes.cdll.LoadLibrary('./LibGPU.so')


# Preparando para el uso de "double ScalGPU(double *x, const double alpha, const int n, const int ThpBlk)"
# Describir la interfaz de la funcion ScalGPU, y asignarla a un objeto en Python
# .restype  se refiere al tipo de lo que retorna la funcion. Si es void valor poner None
# .argtypes se refiere al tipo de los argumentos de la funcion
ESCALA = lib.ScalGPU
ESCALA.restype  = ctypes.c_int
ESCALA.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double, ctypes.c_int, ctypes.c_int]


# Tamano del vector, hilos por bloque y valor de alpah
n     = 4096
ThpBlk= 1024
alpha = 1.5


# Crear el vector de datos (a) y copiarlo a otro (b)
a = np.random.rand(n).astype('d')
b = np.copy(a)

# Llamar a la funcion ScalGPU de "cuda"
error=ESCALA(a, alpha, a.size, ThpBlk)

if(error < 0):
  print('No hay GPUs compatible CUDA '+ str(error))
else:
  for i in range(b.size):
    b[i]=b[i]*alpha
  print('El error es '+ str(LA.norm(a-b)))
