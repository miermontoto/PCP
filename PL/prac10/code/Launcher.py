import os
import sys
import time
from PIL import Image
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from numpy import linalg as LA

debug = "debug" in sys.argv
binarizar = "bin" in sys.argv
diffs = "diffs" in sys.argv
mode = "SECUENCIAL" if os.environ.get("OMP_NUM_THREADS") == '1' else "PARALELO"
cuda = "tpb" in sys.argv

validCalls = {
    'prof': {
        'function': 'mandelProf',
        'name': 'fractalProf',
        'average': 'mediaProf',
        'binary': 'binarizaProf'
    },
    'py': {
        'function': 'mandelPy',
        'name': 'fractalPy',
        'average': 'mediaPy',
        'binary': 'binarizaPy'
    },
    'own': {
        'function': 'mandelAlumnx',
        'name': 'fractalAlumnx',
        'average': 'mediaAlumnx',
        'binary': 'binarizaAlumnx'
    }
}

calls = []
sizes = []

for key in list(validCalls.keys()):
    if key in sys.argv:
        calls.append(validCalls.get(key))
    elif f"-{key}" in sys.argv and validCalls.get(key) in calls:
        calls.remove(validCalls.get(key))

if "sizes" in sys.argv:
    for i in range(sys.argv.index("sizes") + 1, len(sys.argv)):
        try: sizes.append(int(sys.argv[i]))
        except: break

if len(sizes) == 0: sizes.append(4) # marcar error si no se detectan tamaños

if cuda:
    tpb = int(sys.argv[sys.argv.index("tpb") + 1])
    mode = "GPU"
    os.system('make cuda >/dev/null')
else: os.system('make omp >/dev/null')

functions = {
    'mandel': {
        'name': 'mandel',
        'restype': None,
        'argtypes': [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    },
    'media': {
        'name': 'promedio',
        'restype': ctypes.c_double,
        'argtypes': [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    },
    'binariza': {
        'name': 'binariza',
        'restype': None,
        'argtypes': [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double]
    }
}

owners = ['Prof', 'Alumnx']
for owner in owners:
    lib = ctypes.cdll.LoadLibrary(f"./{'cuda' if cuda else 'openmp'}/mandel{owner}{'GPU' if cuda else ''}.so")
    for key, value in functions.items():
        locals()[f"{key}{owner}"] = getattr(lib, f"{value['name']}{'GPU' if cuda else ''}")
        locals()[f"{key}{owner}"].restype = value['restype']
        locals()[f"{key}{owner}"].argtypes = value['argtypes']


# Función de cálculo del fractal en Python
def mandelPy(xmin, ymin, xmax, ymax, maxNoneiter, xres, yres, A):
    if xres > 2048 or yres > 2048: raise Exception("Tamaño de imagen demasiado grande")

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


if __name__ == "__main__":
    xmin = float(sys.argv[1])
    xmax = float(sys.argv[2])
    ymin = float(sys.argv[3])
    maxiter = int(sys.argv[4])
    ymax = xmax - xmin + ymin

    if not "noheader" in sys.argv: print(f"Function;Mode;Size;Time;Error;Average;Average Time{f';Bin (err);Bin Time' if binarizar else ''}")

    for size in sizes:
        yres = size
        xres = size

        for call in calls:
            function = call['function']
            name = call['name']
            averageFunc = call['average']
            binaryFunc = call['binary']
            original = calls[0]['name']

            # Como indicado en clase, tamaños superiores a 2048 suponen un calculo
            # demasiado largo y no son útiles para la práctica.
            # Para poder enviar todos los tamaños en una sola ejecución, se comprueba
            # el tamaño aquí.
            if function == "mandelPy" and size > 2048: continue

            locals()[name] = np.zeros(yres*xres).astype(np.double) # reservar memoria

            # ejecutar función
            calcTime = time.time()
            if cuda: locals()[function](xmin, ymin, xmax, ymax, maxiter, xres, yres, locals()[name], tpb)
            else: locals()[function](xmin, ymin, xmax, ymax, maxiter, xres, yres, locals()[name])
            calcTime = time.time() - calcTime

            # calcular promedio y error
            averageTime = time.time()
            if cuda: average = locals()[averageFunc](xres, yres, locals()[name], tpb)
            else: average = locals()[averageFunc](xres, yres, locals()[name]) # calcular promedio
            averageTime = time.time() - averageTime
            error = "-" if original == name else LA.norm(locals()[name] - locals()[original]) # calcular error

            # imprimir resultados
            print(f"{function};{mode};{size};{calcTime:1.5E};{error};{average};{averageTime:1.5E}", end="" if binarizar else "\n")

            # guardar imágenes
            if debug:
                grabar(locals()[name], xres, yres, f"{name}_{size}.bmp") # guardar archivo
                if diffs and i > 0: grabar(diffImage(locals()[name], locals()[original]), xres, yres, f"diff_{name}_{size}.bmp")

            # binarizar
            if binarizar:
                binName = f"bin_{name}"
                binOriginal = f"bin_{original}"
                locals()[binName] = np.copy(locals()[name]) # copiar imagen para evitar sobreescribirla

                # calcular binarización
                binarizaTime = time.time()
                if cuda: locals()[binaryFunc](xres, yres, locals()[binName], average, tpb)
                else: locals()[binaryFunc](yres, xres, locals()[binName], average)
                binarizaTime = time.time() - binarizaTime

                # calcular e imprimir error
                error = "-" if binName == binOriginal else LA.norm(locals()[binName] - locals()[binOriginal])
                print(f";{error};{binarizaTime:1.5E}")

                # guardar binarizado
                if debug: grabar(locals()[binName], xres, yres, f"{binName}_{size}.bmp")
