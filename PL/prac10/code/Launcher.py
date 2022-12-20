import os
import sys
import time
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from numpy import linalg as LA
from mandel import *

debug = "debug" in sys.argv
binarizar = "bin" in sys.argv
diffs = "diffs" in sys.argv
times = "times" in sys.argv
onlytimes = "onlytimes" in sys.argv

if "tpb" in sys.argv: # Detección de modo CUDA
    try: tpb = int(sys.argv[sys.argv.index("tpb") + 1])
    except:
        tpb = 32
        print("Error al obtener el número de hilos por bloque, se utiliza valor por defecto (32)")
    mode = "GPU"
    cuda = True
    translated = 'cuda'
else:
    mode = "SECUENCIAL" if os.environ.get("OMP_NUM_THREADS") == '1' else "PARALELO"
    cuda = False
    translated = 'omp'

os.system(f"make {translated} >/dev/null") # Se ignoran los mensajes pero no los errores

validFunctions = {
    'omp': {
        'mandel': {
            'normal': 'mandel_normal',
            'collapse': 'mandel_collapse',
            'tasks': 'mandel_tasks',
            'schedule_auto': 'mandel_schedule_auto',
            'schedule_static': 'mandel_schedule_static',
            'schedule_guided': 'mandel_schedule_guided',
            'schedule_dynamic': 'mandel_schedule_dynamic'
        },
        'promedio': {
            'normal': 'promedio_normal',
            'int': 'promedio_int',
            'schedule': 'promedio_schedule',
            'atomic': 'promedio_atomic',
            'critical': 'promedio_critical',
            'vect': 'promedio_vect'
        }
    },
    'cuda': {
        'mandel': {
            'normal': 'mandelGPU_normal',
            'heter': 'mandelGPU_heter',
            'unified': 'mandelGPU_unified',
            'pinned': 'mandelGPU_pinned',
            '1D': 'mandelGPU_1D'
        },
        'promedio': {
            'api': 'promedioGPU_api',
            'shared': 'promedioGPU_shared',
            'param': 'promedioGPU_param',
            'atomic': 'promedioGPU_atomic',
        }
    }
}

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
    'own': {}
}

calls = []
sizes = []

for key in list(validCalls.keys()):
    if key in sys.argv:
        if key == 'own':
            averages = [next(iter(validFunctions[translated]['promedio'].values()))]

            if "averages" in sys.argv:
                if "all" == sys.argv[sys.argv.index("averages") + 1]:
                    averages = list(validFunctions[translated]['promedio'].values())
                else:
                    averages = []
                    for i in range(sys.argv.index("averages") + 1, len(sys.argv)):
                        if sys.argv[i] in validFunctions[translated]['promedio']:
                            averages.append(validFunctions[translated]['promedio'][sys.argv[i]])
                        else: break

            if "methods" in sys.argv:
                if "all" == sys.argv[sys.argv.index("methods") + 1]:
                    for key, value in validFunctions[translated]['mandel'].items():
                        for average in averages:
                            calls.append({
                                'function': value,
                                'name': f'fractalAlumnx{key.capitalize()}',
                                'average': average,
                                'binary': 'binarizaAlumnx'
                            })
                else:
                    for i in range(sys.argv.index("methods") + 1, len(sys.argv)):
                        if sys.argv[i] in validFunctions[translated]['mandel']:
                            for average in averages:
                                calls.append({
                                    'function': validFunctions[translated]['mandel'][sys.argv[i]],
                                    'name': f'fractalAlumnx{sys.argv[i].capitalize()}',
                                    'average': average,
                                    'binary': 'binarizaAlumnx'
                                })
                        else: break
            else:
                for average in averages:
                    calls.append({
                        'function': next(iter(validFunctions[translated]['mandel'].values())),
                        'name': 'fractalAlumnx',
                        'average': average,
                        'binary': 'binarizaAlumnx'
                    })
        else: calls.append(validCalls.get(key))
    elif f"-{key}" in sys.argv:
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

    if not "noheader" in sys.argv:
        base = "Function;Mode;Size;Time"
        if cuda: base += ";TPB"
        if not onlytimes:
            base += ";Error;Average Function;Average"
            if times: base += ";Average Time"
            if binarizar:
                base += ";Binary (err)"
                if times:
                    base += ";Binary Time"
        print(base)

    if cuda: # heat up cache
        for i in range(0, 3):
            size = next(iter(sizes))
            for call in calls:
                locals()[call['name']] = np.zeros(size*size).astype(np.double)
                locals()[call['function']](xmin, ymin, xmax, ymax, maxiter, size, size, locals()[call['name']], tpb)
                locals()[call['average']](size, size, locals()[call['name']], tpb)

    for size in sizes:
        yres = size
        xres = size

        for call in calls:
            function = call['function']
            name = call['name']
            averageFunc = call['average']
            binaryFunc = call['binary']
            original = calls[0]['name']

            checkCuda = cuda and not "Py" in function

            # Como indicado en clase, tamaños superiores a 2048 suponen un calculo
            # demasiado largo y no son útiles para la práctica.
            # Para poder enviar todos los tamaños en una sola ejecución, se comprueba
            # el tamaño aquí.
            if "Py" in function or "Py" in name and size > 2048: continue

            locals()[name] = np.zeros(yres*xres).astype(np.double) # reservar memoria

            # ejecutar función
            calcTime = time.time()
            if checkCuda: locals()[function](xmin, ymin, xmax, ymax, maxiter, xres, yres, locals()[name], tpb)
            else: locals()[function](xmin, ymin, xmax, ymax, maxiter, xres, yres, locals()[name])
            calcTime = time.time() - calcTime

            # calcular promedio y error
            averageTime = time.time()
            if checkCuda: average = locals()[averageFunc](xres, yres, locals()[name], tpb)
            else: average = locals()[averageFunc](xres, yres, locals()[name]) # calcular promedio
            averageTime = time.time() - averageTime
            try: error = "-" if original == name else LA.norm(locals()[name] - locals()[original]) # calcular error
            except: error = "NaN"

            if cuda: tpbStr = "-" if "Py" in function else tpb

            # imprimir resultados
            results = f"{function};{mode};{size};{calcTime:1.5E}"
            if cuda: results += f";{tpbStr}"
            if not onlytimes:
                results += f";{error};{averageFunc};{average}"
                if times: results += f";{averageTime:1.5E}"
            print(results, end="" if binarizar else "\n")

            # guardar imágenes
            if debug:
                grabar(locals()[name], xres, yres, f"{name}_{size}.bmp") # guardar archivo
                if diffs and i > 0: grabar(diffImage(locals()[name], locals()[original]), xres, yres, f"diff_{name}_{size}.bmp")

            # binarizar
            if binarizar and not onlytimes:
                binName = f"bin_{name}"
                binOriginal = f"bin_{original}"
                locals()[binName] = np.copy(locals()[name]) # copiar imagen para evitar sobreescribirla

                # calcular binarización
                binarizaTime = time.time()
                if checkCuda: locals()[binaryFunc](xres, yres, locals()[binName], average, tpb)
                else: locals()[binaryFunc](yres, xres, locals()[binName], average)
                binarizaTime = time.time() - binarizaTime

                # calcular e imprimir error
                error = "-" if binName == binOriginal else LA.norm(locals()[binName] - locals()[binOriginal])
                print(f";{error};{f'{binarizaTime:1.5E}' if times else ''}")

                # guardar binarizado
                if debug: grabar(locals()[binName], xres, yres, f"{binName}_{size}.bmp")
