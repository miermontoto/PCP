import ctypes
import numpy as np
import random
import time
import sys
import math
from numpy.ctypeslib import ndpointer as ND
from numpy           import linalg    as LA

list = ['IccO0', 'IccO3']
funcs = ['MyDGEMMT']
validFuncs = ['MyDGEMMT', 'MyDGEMM', 'MyDGEMMB']

for func in validFuncs:
  if func in sys.argv: funcs = [func]

for func in funcs:
  if not func in validFuncs: raise Exception('Invalid function: ' + func)


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

simplifyOutput = "simplify" in sys.argv
calculatePython = "python" in sys.argv

Normal = 1
TransA = 2
acceptedTypes = [Normal, TransA]

talla = [1000, 2000, 3000]
rept  = [10, 8, 6, 4, 2]
alpha = 1.3
beta  = 1.7
tipos = [TransA]

if "test" in sys.argv:
  for i in range(0, len(talla)):
    talla[i] /= 10

if not simplifyOutput:
  print("Function;Library;Size;Repetitions;Time;Error;Type;BlockSize")

for tipo in tipos:
  if tipo not in acceptedTypes: raise Exception(f"Invalid type {tipo}")
  for func in funcs:
    if simplifyOutput: print(f"{func}...")
    validMeasurement = True
    for i in range(0, len(talla)):
      m = talla[i]
      n = m + 1
      k = m - 1
      blk = int(m / 10)
      if "squareSize" in sys.argv:
        n = m
        k = m

      A = np.random.rand(m, k).astype(np.float64)
      B = np.random.rand(k, n).astype(np.float64)
      C = np.random.rand(m, n).astype(np.float64)
      D = np.copy(C)
      D = beta*D + alpha*(A @ B)

      if not simplifyOutput and calculatePython:
        start = time.time()
        for j in range(rept[i]):
          D = np.copy(C)
          D = beta*D + alpha*(A @ B)
        print(f"Python;numpy;{m}x{n}x{k};{rept[i]};{time.time() - start};-;-;-")

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
          print(f"{func};{lib};{m}x{n}x{k};{rept[i]};{time.time() - start};{errmedio / rept[i]};{tipo};{blk if func == 'MyDGEMMB' else '-'}")
        validMeasurement &= (errmedio/(rept[i]) < 1.0E-10)
    if simplifyOutput:
      print("OK" if validMeasurement else "ERROR")
  print()

