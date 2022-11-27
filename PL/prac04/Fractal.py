import sys
import os
from PIL import Image
import ctypes
import numpy as np
from numpy import linalg as LA
from numpy.ctypeslib import ndpointer
import time


debug = "debug" in sys.argv
binarizar = "binarizar" in sys.argv
diffs = "diffs" in sys.argv
tiempos = "tiempos" in sys.argv
mode = "SECUENCIAL" if os.environ.get("OMP_NUM_THREADS") == '1' else "PARALELO"

validCalls = ['mandelProf', 'mandelPy', 'mandelAlumnx']
assignedNames = ['fractalProf', 'fractalPy', 'fractalAlumnx']
assignedAverages = ['mediaProf', 'mediaPy', 'mediaAlumnx']
assignedBinaries = ['binarizaProf', 'binarizaPy', 'binarizaAlumnx']

calls = []
names = []
averages = []
binaries = []
sizes = []

for i in range(len(validCalls)):
    if validCalls[i] in sys.argv:
        calls.append(validCalls[i])
        names.append(assignedNames[i])
        averages.append(assignedAverages[i])
        binaries.append(assignedBinaries[i])
    elif sys.argv[i][1:] in validCalls and sys.argv[i][0] == "-" and sys.argv[i][1:] in calls:
        calls.remove(sys.argv[i][1:])
        names.remove(assignedNames[validCalls.index(sys.argv[i][1:])])
        averages.remove(assignedAverages[validCalls.index(sys.argv[i][1:])])
        binaries.remove(assignedBinaries[validCalls.index(sys.argv[i][1:])])

if "sizes" in sys.argv:
    for i in range(sys.argv.index("sizes"), len(sys.argv)):
        try: sizes.append(int(sys.argv[i]))
        except: pass

if len(sizes) == 0: sizes.append(4) # marcar error si no se detectan tamaños

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

mediaAlumnx = libAlumnx.promedio_atomic
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

    if not "noheader" in sys.argv: print(f"Function;Mode;Size;Time;Error;Average{';Average Time' if tiempos else ''}{f';Bin (err)' if binarizar else ''}{f';Bin Time' if binarizar and tiempos else ''}")

    for size in sizes:
        yres = size
        xres = size

        for name in names: locals()[name] = np.zeros(yres*xres).astype(np.double) # reservar memoria

        for i in range(len(calls)):
            calcTime = time.time()
            locals()[calls[i]](xmin, ymin, xmax, ymax, maxiter, xres, yres, locals()[names[i]]) # ejecutar función
            calcTime = time.time() - calcTime

            averageTime = time.time()
            average = locals()[averages[i]](xres, yres, locals()[names[i]]) # calcular promedio
            averageTime = time.time() - averageTime
            averageStr = '-' if average == 0.0 else average

            error = "-" if i == 0 else LA.norm(locals()[names[i]] - locals()[names[0]]) # calcular error
            errorStr = '-' if str(average)[0] == '-' else error

            print(f"{calls[i]};{mode};{size};{calcTime};{errorStr};{averageStr}{f';{averageTime:1.5E}' if tiempos else ''}", end="" if binarizar else "\n")

            if debug:
                grabar(locals()[names[i]], xres, yres, f"{names[i]}_{size}.bmp") # guardar archivo
                if diffs and i > 0: grabar(diffImage(locals()[names[i]], locals()[names[0]]), xres, yres, f"diff_{names[i]}_{size}.bmp")

            if binarizar:
                locals()[f"bin{names[i]}"] = np.copy(locals()[names[i]])

                binarizationTime = time.time()
                locals()[binaries[i]](yres, xres, locals()[f"bin{names[i]}"], average)
                binarizationTime = time.time() - binarizationTime

                error = "-" if i == 0 else LA.norm(locals()[f"bin{names[i]}"] - locals()[f"bin{names[0]}"])
                print(f";{error}{f';{binarizationTime:1.5E}' if tiempos else ''}")
                if debug: grabar(locals()[f"bin{names[i]}"], xres, yres, f"bin_{names[i]}_{size}.bmp")

        with open(f"{size}.done", "w") as f: f.write(time.strftime("%H:%M:%S", time.localtime()))

    # remove all done files
    for size in sizes: os.remove(f"{size}.done")
