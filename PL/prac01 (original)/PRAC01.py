import ctypes
import numpy  as np 

from numpy.ctypeslib import ndpointer as ND
from numpy           import linalg    as LA
from random          import random     
from time            import time

LibGccO0 = ctypes.cdll.LoadLibrary('LIBS/PRACGccO0.so')
LibGccO3 = ctypes.cdll.LoadLibrary('LIBS/PRACGccO3.so')
LibIccO0 = ctypes.cdll.LoadLibrary('LIBS/PRACIccO0.so')
LibIccO3 = ctypes.cdll.LoadLibrary('LIBS/PRACIccO3.so')

GccO0 = LibGccO0.MyDGEMM
GccO3 = LibGccO3.MyDGEMM
IccO0 = LibIccO0.MyDGEMM
IccO3 = LibIccO3.MyDGEMM

GccO0.restype = ctypes.c_double
GccO3.restype = ctypes.c_double
IccO0.restype = ctypes.c_double
IccO3.restype = ctypes.c_double

GccO0.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]
GccO3.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]
IccO0.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]
IccO3.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]

talla = [200, 250, 300, 350, 400]
rept  = [ 10,   8,   6,   4,   2]

alpha = 1.3
beta  = 1.7
tipo  = 1  # 1 normal, 2 transpuesta de A

for i in range(0,len(talla)):
   m      = talla[i]
   n      = m + int(m/2)
   k      = m - int(m/2)

   A = np.random.rand(m, k).astype(np.float64)
   B = np.random.rand(k, n).astype(np.float64)
   C = np.random.rand(m, n).astype(np.float64)

   D = np.copy(C)
   secs = time()
   for j in range(rept[i]):
     D = beta*D + alpha*(A @ B)
   TIEMPO = (time()- secs)/rept[i]
   print(f"Python {m}x{n}x{k} Segundos={TIEMPO:1.5E}")

   A = np.asarray(A, order='F')
   B = np.asarray(B, order='F')

   F = np.asarray(C, order='F')
   secs = time()
   for j in range(rept[i]):
     TiempC=GccO0(tipo, m, n, k, alpha, A, m, B, k, beta, F, m)
   TIEMPO = (time()- secs)/rept[i]
   print(f"GccO0  {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")
   print(f"Error entre Python y GccO0 {LA.norm(D-F, 'fro'):1.5E}")

   F = np.asarray(C, order='F')
   secs = time()
   for j in range(rept[i]):
     TiempC=GccO3(tipo, m, n, k, alpha, A, m, B, k, beta, F, m)
   TIEMPO = (time()- secs)/rept[i]
   print(f"GccO3  {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")
   print(f"Error entre Python y GccO3 {LA.norm(D-F, 'fro'):1.5E}")

   F = np.asarray(C, order='F')
   secs = time()
   for j in range(rept[i]):
     TiempC=IccO0(tipo, m, n, k, alpha, A, m, B, k, beta, F, m)
   TIEMPO = (time()- secs)/rept[i]
   print(f"IccO0  {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")
   print(f"Error entre Python y IccO0 {LA.norm(D-F, 'fro'):1.5E}")

   F = np.asarray(C, order='F')
   secs = time()
   for j in range(rept[i]):
     TiempC=IccO3(tipo, m, n, k, alpha, A, m, B, k, beta, F, m)
   TIEMPO = (time()- secs)/rept[i]
   print(f"IccO3  {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")
   print(f"Error entre Python y IccO3 {LA.norm(D-F, 'fro'):1.5E}\n")
