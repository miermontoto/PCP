import sys
import os
from PIL import Image
import ctypes
import numpy as np
from numpy import linalg as LA
from numpy.ctypeslib import ndpointer
from time import time

def printTime(startTime, msg):
    print(f"({msg}) time: {time() - startTime}")

debug = "debug" in sys.argv
binarizar = "binarizar" in sys.argv
diffs = "diffs" in sys.argv

validCalls = ['mandelProf', 'mandelPy', 'mandelAlumnx']
assignedNames = ['fractalProf', 'fractalPy', 'fractalAlumnx']
assignedAverages = ['mediaProf', 'mediaPy', 'mediaAlumnx']
assignedBinaries = ['binarizaProf', 'binarizaPy', 'binarizaAlumnx']
assignedBinariesNames = ['bin_fractalProf', 'bin_fractalPy', 'bin_fractalAlumnx']

calls = []
names = []
averages = []
binaries = []
binariesNames = []
sizes = []
for i in range(5, len(sys.argv)):
    if sys.argv[i] in validCalls:
        calls.append(sys.argv[i])
        names.append(assignedNames[validCalls.index(sys.argv[i])])
        averages.append(assignedAverages[validCalls.index(sys.argv[i])])
        if binarizar:
            binaries.append(assignedBinaries[validCalls.index(sys.argv[i])])
            binariesNames.append(assignedBinariesNames[validCalls.index(sys.argv[i])])
    if "sizes" in sys.argv[i]:
        for j in range(i+1, len(sys.argv)):
            try:
                sizes.append(int(sys.argv[j]))
            except:
                pass

if sizes == []: sizes.append(4)

# obtain environment variable "OMP_NUM_THREADS"
mode = "SECUENCIAL" if os.environ.get("OMP_NUM_THREADS") == '1' else "PARALELO"

# Prepara gestión librería externa de Profesor
libProf = ctypes.cdll.LoadLibrary('./mandelProf.so')

mandelProf = libProf.mandel
mandelProf.restype  = None
mandelProf.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

mediaProf = libProf.promedio
mediaProf.restype  = ctypes.c_double
mediaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

binarizaProf = libProf.binariza
binarizaProf.restype  = None
binarizaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double]


# Preparar gestión librería externa de Alumnx llamada mandelAlumnx.so
libAlumnx = ctypes.cdll.LoadLibrary('./mandelAlumnx.so')

mandelAlumnx = libAlumnx.mandel
mandelAlumnx.restype  = mandelProf.restype
mandelAlumnx.argtypes = mandelProf.argtypes

mediaAlumnx = libAlumnx.promedio
mediaAlumnx.restype  = mediaProf.restype
mediaAlumnx.argtypes = mediaProf.argtypes

binarizaAlumnx = libAlumnx.binariza
binarizaAlumnx.restype  = binarizaProf.restype
binarizaAlumnx.argtypes = binarizaProf.argtypes


# Función de cálculo del fractal en Python
def mandelPy(xmin, ymin, xmax, ymax, maxNoneiter, xres, yres, A):
    if xres > 2048 or yres > 2048: return

    dx = (xmax - xmin) / xres
    dy = (ymax - ymin) / yres

    for i in range(xres):
        for j in range(yres):
            c = complex(xmin + i * dx, ymin + j * dy)
            z = complex(0, 0)

            k = 1
            while k < maxiter and abs(z) < 2:
                z = z*z + c
                k += 1
            A[i + j * xres] = 0 if k >= maxiter else k

def mediaPy(xres, yres, A):
    return np.mean(A)

def binarizaPy(xres, yres, A, average):
    for i in range(len(A)):
        A[i] = 0 if A[i] < average else 255

# otras funciones auxiliares
def diffImage(vect1, vect2):
    vectResult = np.zeros(vect1.shape)
    for i in range(len(vect1)):
        vectResult[i] = 255 if vect1[i] != vect2[i] else 0
    return vectResult


def grabar(vect, xres, yres, output):
    A2D=vect.astype(np.ubyte).reshape(yres,xres) #row-major por defecto
    im=Image.fromarray(A2D)
    im.save(output)
    #print(f"+{output}")


# main
if __name__ == "__main__":
    xmin = float(sys.argv[1])
    xmax = float(sys.argv[2])
    ymin = float(sys.argv[3])
    maxiter = int(sys.argv[4])

    ymax = xmax - xmin + ymin

    if not "noheader" in sys.argv: print("Function;Mode;Size;Time;Error;Average", end=";Bin (err)\n" if binarizar else "\n")

    for size in sizes:
        yres = size
        xres = size

        for name in names: locals()[name] = np.zeros(yres*xres).astype(np.double) # reservar memoria

        for i in range(len(calls)):
            startTime = time()
            locals()[calls[i]](xmin, ymin, xmax, ymax, maxiter, xres, yres, locals()[names[i]]) # ejecutar función
            finishTime = time()

            average = locals()[averages[i]](yres, xres, locals()[names[i]]) # calcular promedio
            error = "-" if i == 0 else LA.norm(locals()[names[i]] - locals()[names[0]]) # calcular error

            print(f"{calls[i]};{mode};{size};{finishTime - startTime};{error};{average}", end="" if binarizar else "\n")

            if debug:
                grabar(locals()[names[i]], xres, yres, f"{names[i]}_{size}.bmp") # guardar archivo
                if diffs and i > 0: grabar(diffImage(locals()[names[i]], locals()[names[0]]), xres, yres, f"diff_{names[i]}_{size}.bmp")

            if binarizar:
                locals()[binariesNames[i]] = np.copy(locals()[names[i]])

                locals()[binaries[i]](yres, xres, locals()[binariesNames[i]], average)
                error = "-" if i == 0 else LA.norm(locals()[binariesNames[i]] - locals()[binariesNames[0]])
                print(f";{error}")
                if debug: grabar(locals()[binariesNames[i]], xres, yres, f"{binariesNames[i]}_{size}.bmp")
