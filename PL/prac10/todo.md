# todo
- [x] Código
    - [x] prac4
    - [x] prac6
- [x] Recopilación de resultados y tiempos
    - [x] prac4
    - [x] prac6
- [x] Informe

## prac4
- [x] Código Python
- [x] Código OpenMP
- [x] Python: test promedios

### py
- [x] Funciones de cálculo básico (mandel, promedio, binarizado)
- [x] Rescritura del código original
- [x] Lectura de funciones desde parámetros
- [x] Lectura de tamaños desde parámetros
- [x] Lectura de características desde parámetros
- [x] Incluir tiempos de ejecución de funciones no principales (promedio, binarizado)

### mandel
- [x] Parallel for
- [x] Collapse
- [x] Tasks
- [x] Schedule (comprobar static, dynamic (varios), guided)

### promedio
- [x] Reduction
    Punto de comparación con el resto.
- [x] Critical
    Debería ser más rápido que reduction.
- [x] Atomic
    Debería ser más rápido que critical: usa recursos HW específicos y no hay candados.
- [x] Vectorization
    Debería ser el más rápido: elimina el overhead de atomic y critical.
- [x] Reduction (int)
    La mitad de rápido que el reduction normal debido a los casteos.
- [x] Reduction + schedule(static)
    Debería ser más rápido que el reduction normal al tratar de evitar False Sharing.

### binarizado
- [x] Básico

## prac6
- [x] Código Python
    - [x] Lectura de parámetros de GPU
    - [x] Llamadas a CUDA
    - [x] Combinar con launcher original (prac4)
    - [x] Ejecución de múltiples funciones por librería alumnx.
    - [x] Combinar launcher de promedios en el Launcher.
- [x] Código CUDA
- [x] Makefiles

### mandel
- [x] Básico
- [x] Pinned memory
- [x] Unified memory
- [x] Cálculo heterogéneo (con unified)
- [x] 1D

### promedio
- [x] API
- [x] Kernel (con shared)
- [x] Kernel (sin shared, con paso de parámetros)
- [x] Atomic

### binarizado
- [x] Básico
