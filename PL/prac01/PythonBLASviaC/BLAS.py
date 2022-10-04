import ctypes
import numpy  as np 

from numpy.ctypeslib import ndpointer as ND
from numpy           import linalg    as LA
from random          import random     
from time            import time

lib = ctypes.cdll.LoadLibrary('./LibBLAS.so')

DGEMMRow = lib.MyDGEMMRow
DGEMMCol = lib.MyDGEMMCol

DGEMMRow.restype  = ctypes.c_double
DGEMMCol.restype  = ctypes.c_double

DGEMMRow.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                     ND(ctypes.c_double, flags="C"), ctypes.c_int, ND(ctypes.c_double, flags="C"), ctypes.c_int, ctypes.c_double, 
                     ND(ctypes.c_double, flags="C"), ctypes.c_int]

DGEMMCol.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                     ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double, 
                     ND(ctypes.c_double, flags="F"), ctypes.c_int]

talla = [2000, 2500, 3000, 3500, 4000]
rept  = [   6,    5,    4,    3,    2]

alpha = 1.3
beta  = 0.7

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
   print(f"Python  @  {m}x{n}x{k} Segundos={TIEMPO:1.5E}")

   F = np.copy(C)
   secs = time()
   for j in range(rept[i]):
     TiempC=DGEMMRow(m, n, k, alpha, A, k, B, n, beta, F, n)
   TIEMPO = (time()- secs)/rept[i]
   TiempC = TiempC
   print(f"MyDGEMMRow {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")

   A = np.asarray(A, order='F')
   B = np.asarray(B, order='F')
   G = np.asarray(C, order='F')
   secs = time()
   for j in range(rept[i]):
     TiempC=DGEMMCol(m, n, k, alpha, A, m, B, k, beta, G, m)
   TIEMPO = (time()- secs)/rept[i]
   TiempC = TiempC
   print(f"MyDGEMMCol {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")

   print(f"Error Python vs. MyDGEMMRow {LA.norm(D-F, 'fro'):1.5E}")
   print(f"Error Python vs. MyDGEMMCol {LA.norm(D-G, 'fro'):1.5E}")
   print(f"Error C Rows vs. C Cols     {LA.norm(G-F, 'fro'):1.5E}\n")
   