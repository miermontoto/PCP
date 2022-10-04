import numpy as np

from numpy  import linalg as LA
from random import random

m = 80
n = int(m/2)

A = np.random.rand(m, n).astype(np.float64)
B = np.random.rand(n, m).astype(np.float64)

A2D = np.copy(A)    # copia A en A2D
B2D = np.copy(B).T  # copia B transpuesta en B2D
C2D = A2D+B2D       # suma elemento a elemento


B1D = np.copy(B).reshape(m*n, order='F') # copia B en el vector B1D usandor colum-major order
A1D = np.copy(A).reshape(m*n, order='C') # copya A en el vector A1D usandor row-major order (por defecto)
D1D = A1D + B1D                          # suma elemento a elmento
D2D = np.reshape(D1D, (m, n))            # transforma el vector D1D en la matriz D2D usando el order por defecto

print('El error es '+ str(LA.norm(C2D-D2D,'fro'))) # imprime el error
