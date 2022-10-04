import numpy  as np

from numpy  import linalg as LA
from random import random     
from time   import time

def Naive(m, n, k, A, B, C):
  for i in range(m):
    for j in range(n):
      dtmp=0.0
      for p in range(k):
        dtmp = dtmp + A[i,p]*B[p,j]
      C[i,j]=dtmp

talla=[100, 150, 200, 250, 300]
rept1=[ 10,   5,   3,   2,   1]
rept2=[100,  50,  30,  20,  10]

for i in range(0,len(talla)):
   m      = talla[i]
   n      = m + 1
   k      = m - 1

   A = np.random.rand(m, k).astype(np.float64)
   B = np.random.rand(k, n).astype(np.float64)
   C = np.zeros((m, n), dtype=np.float64)
   D = np.zeros((m, n), dtype=np.float64)

   secs = time()
   for j in range(rept1[i]):
     Naive(m, n, k, A, B, C)
   TIEMPO = (time()- secs)/rept1[i]
   print(f"Naive 3L {m}x{n}x{k} Segundos={TIEMPO:1.5E}")

   secs = time()
   for j in range(rept2[i]):
     D = A @ B
   TIEMPO = (time()- secs)/rept2[i]
   print(f"Python @ {m}x{n}x{k} Segundos={TIEMPO:1.5E}")
   
   print(f"Con {m}x{n}x{k} error entre soluciones {LA.norm(C-D, 'fro'):1.5E}\n")
