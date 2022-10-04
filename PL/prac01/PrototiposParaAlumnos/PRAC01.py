import ctypes
import numpy as np
import random
import time
from numpy.ctypeslib import ndpointer as ND
from numpy           import linalg    as LA


list = ['GccO0', 'GccO3', 'IccO0', 'IccO3']

for lib in list:
    Lib = ctypes.cdll.LoadLibrary('LIBS/PRAC' + lib + '.so')
    func = Lib.MyDGEMM
    func.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]
    func.restype = ctypes.c_double

    # save the function for later use
    locals()[lib] = func


talla = [200, 250, 300, 350, 400]
rept  = [ 10,   8,   6,   4,   2]
alpha = random.randint(1, 10) * random.random()
beta  = random.randint(1, 10) * random.random()

for tipo in [1, 2]:
  print('\nTipo de multiplicación: ', 'Normal' if tipo == 1 else 'Transpuesta')
  for i in range(0,len(talla)):
    m = talla[i]
    n = m + int(m/2)
    k = m - int(m/2)

    A = np.random.rand(m, k).astype(np.float64)
    B = np.random.rand(k, n).astype(np.float64)
    C = np.random.rand(m, n).astype(np.float64)

    D = np.copy(C)
    D = beta*D + alpha*(A @ B)

    A = np.asarray(A, order='F')
    B = np.asarray(B, order='F')

    F = np.asarray(C, order='F')
    errmedio = 0.0
    for j in range(rept[i]):
      for lib in list:
        F = np.asarray(C, order='F')
        locals()[lib](tipo, m, n, k, alpha, A, m, B, k, beta, F, m)
        errmedio += LA.norm(D-F, 'fro')

    print(f"Talla: {talla[i]}, error medio: {errmedio/(4*rept[i]):1.5E}")
