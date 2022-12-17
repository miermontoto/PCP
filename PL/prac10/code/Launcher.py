import os
import sys
import time
from PIL import Image
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from numpy import linalg as LA
from mandel import *

debug = "debug" in sys.argv
binarizar = "bin" in sys.argv
diffs = "diffs" in sys.argv
mode = "SECUENCIAL" if os.environ.get("OMP_NUM_THREADS") == '1' else "PARALELO"
cuda = "tpb" in sys.argv

if cuda:
    tpb = int(sys.argv[sys.argv.index("tpb") + 1])
    mode = "GPU"
    os.system('make cuda >/dev/null')
else: os.system('make omp >/dev/null')

translated = 'cuda' if cuda else 'openmp'

validFunctions = {
    'openmp': {
        'mandel': {
            'normal': 'mandel_normal',
            'tasks': 'mandel_tasks',
            'schedule': 'mandel_schedule',
        },
        'promedio': {
            'normal': 'promedio_normal',
            'atomic': 'promedio_atomic',
            'critical': 'promedio_critical',
            'vect': 'promedio_vect',
            'int': 'promedio_int'
        }
    },
    'cuda': {
        'mandel': {
            'normal': 'mandelGPU_normal',
            'heter': 'mandelGPU_heter',
            'unified': 'mandelGPU_unified',
            'pinned': 'mandelGPU_pinned'
        },
        'promedio': {
            'api': 'promedioGPU_api',
            'shared': 'promedioGPU_shared'
        }
    }
}

validCalls = {
    'prof': {
        'function': 'mandelProf',
        'name': 'fractalProf',
        'average': 'mediaProf',
        'binary': 'binarizaProf',
    },
    'py': {
        'function': 'mandelPy',
        'name': 'fractalPy',
        'average': 'mediaPy',
        'binary': 'binarizaPy'
    },
    'own': {}
}

calls = []
sizes = []

for key in list(validCalls.keys()):
    if key in sys.argv:
        if key == 'own':
            targetAverage = next(iter(validFunctions[translated]['promedio'].values()))
            if "average" in sys.argv and sys.argv[sys.argv.index("average") + 1] in validFunctions[translated]['promedio']:
                targetAverage = validFunctions[translated]['promedio'][sys.argv[sys.argv.index("average") + 1]]

            if "methods" in sys.argv:
                if "all" == sys.argv[sys.argv.index("methods") + 1]:
                    for key, value in validFunctions[translated]['mandel'].items():
                        calls.append({
                            'function': value,
                            'name': f'fractalAlumnx{key.capitalize()}',
                            'average': targetAverage,
                            'binary': 'binarizaAlumnx'
                        })
                else:
                    for i in range(sys.argv.index("methods") + 1, len(sys.argv)):
                        if sys.argv[i] in validFunctions[translated]['mandel']:
                            calls.append({
                                'function': validFunctions[translated]['mandel'][sys.argv[i]],
                                'name': f'fractalAlumnx{sys.argv[i].capitalize()}',
                                'average': targetAverage,
                                'binary': 'binarizaAlumnx'
                            })
                        else: break
            else: calls.append({
                'function': next(iter(validFunctions[translated]['mandel'].values())),
                'name': 'fractalAlumnx',
                'average': targetAverage,
                'binary': 'binarizaAlumnx'
            })
        else: calls.append(validCalls.get(key))
    elif f"-{key}" in sys.argv and validCalls.get(key) in calls:
        calls.remove(validCalls.get(key))

if "sizes" in sys.argv:
    for i in range(sys.argv.index("sizes") + 1, len(sys.argv)):
        try: sizes.append(int(sys.argv[i]))
        except: break

if len(sizes) == 0: sizes.append(4) # marcar error si no se detectan tamaños

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
        if owner == 'Alumnx' and (key == 'mandel' or key == 'media'): continue
        locals()[f"{key}{owner}"] = getattr(lib, f"{value['name']}{'GPU' if cuda else ''}")
        locals()[f"{key}{owner}"].restype = value['restype']
        locals()[f"{key}{owner}"].argtypes = value['argtypes']

    if owner == "Alumnx":
        for call in calls:
            if "Prof" in call['function'] or "Py" in call['function']: continue
            locals()[f"{call['function']}"] = getattr(lib, call['function'])
            locals()[f"{call['function']}"].restype = functions['mandel']['restype']
            locals()[f"{call['function']}"].argtypes = functions['mandel']['argtypes']
            locals()[f"{call['average']}"] = getattr(lib, call['average'])
            locals()[f"{call['average']}"].restype = functions['media']['restype']
            locals()[f"{call['average']}"].argtypes = functions['media']['argtypes']

if __name__ == "__main__":
    xmin = float(sys.argv[1])
    xmax = float(sys.argv[2])
    ymin = float(sys.argv[3])
    maxiter = int(sys.argv[4])
    ymax = xmax - xmin + ymin

    if not "noheader" in sys.argv: print(f"Function;Mode;Size;{'TPB;' if cuda else ''}Time;Error;Average;Average Time{f';Bin (err);Bin Time' if binarizar else ''}")

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
            print(f"{function};{mode};{size};{f'{tpb};' if cuda else ''}{calcTime:1.5E};{error};{average};{averageTime:1.5E}", end="" if binarizar else "\n")

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
