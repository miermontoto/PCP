# todo
- [ ] Código
    - [ ] prac4
    - [ ] prac6
- [ ] Recopilación de resultados y tiempos
    - [ ] prac4
    - [ ] prac6
- [ ] Informe
    - [ ] prac4
    - [ ] prac6

## prac4
- [x] Código Python
- [ ] Código OpenMP
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
- [ ] Reduction + schedule(static)
    Debería ser más rápido que el reduction normal al tratar de evitar False Sharing.

### binarizado
- [x] Básico

## prac6
- [x] Código Python
    - [x] Lectura de parámetros de GPU
    - [x] Llamadas a CUDA
    - [x] Combinar con launcher original (prac4)
    - [x] Ejecución de múltiples funciones por librería alumnx.
    - [ ] Arreglar launcher de promedios
- [ ] Código CUDA
- [x] Makefiles

### mandel
- [x] Básico
- [x] Pinned memory
- [x] Unified memory
- [ ] Cálculo heterogéneo (con unified)
- [ ] 1D
- [ ] 3D

### promedio
- [x] API
- [x] Kernel (con shared)
- [ ] Kernel (sin shared, con paso de parámetros)

### binarizado
- [x] Básico
