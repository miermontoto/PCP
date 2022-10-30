import ctypes
import numpy as np
import random
import time
import math
from numpy.ctypeslib import ndpointer as ND
from numpy           import linalg    as LA

def printTime(startTime, msg):
    print(f"({msg}) time: {time.time() - startTime}")

list = ['IccO3']
funcs = ['MyDGEMM', 'MyDGEMMT', 'MyDGEMMB']

for lib in list:
    Lib = ctypes.cdll.LoadLibrary('LIBS/PRAC' + lib + '.so')
    for func in funcs:
      var = Lib.__getattr__(func)
      var.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int]
      if func == 'MyDGEMMB': var.argtypes.append(ctypes.c_int)
      var.restype = ctypes.c_double

      # save the function for later use
      locals()[f"{lib}_{func}"] = var


print()
pythonMeasuring = False
simplifyOutput = True

talla = [200, 250, 300, 350, 400]
rept  = [ 10,   8,   6,   4,   2]
alpha = random.randint(1, 10) * random.random()
beta  = random.randint(1, 10) * random.random()
tipos = [1, 2]

for tipo in tipos:
  if len(tipos) > 1: print(f"Tipo: {tipo}")
  for func in funcs:
    print(func, end='\n' if not simplifyOutput else '... ')
    validMeasurement = True
    for i in range(0, len(talla)):
      m = talla[i]
      n = m
      k = m  # matrices cuadradas
      blk = int(m / 10)

      if not simplifyOutput: print(f"Talla: {m}, repeticiones: {rept[i]}")

      A = np.random.rand(m, k).astype(np.float64)
      B = np.random.rand(k, n).astype(np.float64)
      C = np.random.rand(m, n).astype(np.float64)
      D = np.copy(C)
      D = beta*D + alpha*(A @ B)

      if pythonMeasuring:
        start = time.time()
        for j in range(rept[i]):
          D = np.copy(C)
          D = beta*D + alpha*(A @ B)
        printTime(start, 'Python')

      A = np.asarray(A, order='F')
      B = np.asarray(B, order='F')

      F = np.asarray(C, order='F')
      for lib in list:
        start = time.time()
        errmedio = 0.0
        for j in range(rept[i]):
          F = np.asarray(C, order='F')
          if func == 'MyDGEMMB':
            locals()[f"{lib}_{func}"](tipo, m, n, k, alpha, A, m, B, k, beta, F, m, blk)
          else: locals()[f"{lib}_{func}"](tipo, m, n, k, alpha, A, m, B, k, beta, F, m)
          errmedio += LA.norm(D-F, 'fro')

        if not simplifyOutput:
          printTime(start, f"{lib}, e: {errmedio/(rept[i]):1.5E}")
        validMeasurement &= (errmedio/(rept[i]) < 1.0E-10)
    if simplifyOutput:
      print("OK" if validMeasurement else "ERROR")
  print()

