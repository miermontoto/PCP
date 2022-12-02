import sys
import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import time

mode = "SECUENCIAL" if os.environ.get("OMP_NUM_THREADS") == '1' else "PARALELO"

validCalls = ['int', 'critical', 'reduction', 'atomic', 'vectorization']
calls = []
sizes = []

if "all" in sys.argv:
    for call in validCalls:
        calls.append(f"promedio_{call}")
else:
    for i in range(len(validCalls)):
        if validCalls[i] in sys.argv:
            calls.append(f"promedio_{validCalls[i]}")

if "sizes" in sys.argv:
    for i in range(sys.argv.index("sizes"), len(sys.argv)):
        try: sizes.append(int(sys.argv[i]))
        except: pass

if len(sizes) == 0: sizes.append(4) # marcar error si no se detectan tamaños

libProf = ctypes.cdll.LoadLibrary('./mandelProf.so')
libAlumnx = ctypes.cdll.LoadLibrary('./mandelAlumnx.so')

mandel = libAlumnx.mandel
mandel.restype  = None
mandel.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

mediaProf = libProf.promedio
mediaProf.restype  = ctypes.c_double
mediaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

for call in calls:
    locals()[call]          = libAlumnx.__getattr__(call)
    locals()[call].restype  = ctypes.c_double
    locals()[call].argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# main
if __name__ == "__main__":
    xmin = float(sys.argv[1])
    xmax = float(sys.argv[2])
    ymin = float(sys.argv[3])
    maxiter = int(sys.argv[4])

    ymax = xmax - xmin + ymin

    if not "noheader" in sys.argv: print("Average;Mode;Size;Time;Error")

    for size in sizes:
        yres = size
        xres = size

        fractal = np.zeros(yres*xres).astype(np.double) # reservar memoria
        mandel(xmin, ymin, xmax, ymax, maxiter, xres, yres, fractal) # calcular con la librería de Profesor

        profTime = time.time()
        mediaZero = mediaProf(xres, yres, fractal)
        profTime = time.time() - profTime
        print(f"promedio_prof;{mode};{size};{profTime:1.5E};-")

        for i in range(len(calls)):
            calcTime = time.time()
            average = locals()[calls[i]](xres, yres, fractal) # ejecutar función
            calcTime = time.time() - calcTime

            error = abs(average - mediaZero) # calcular error

            print(f"{calls[i]};{mode};{size};{calcTime:1.5E};{error}")
