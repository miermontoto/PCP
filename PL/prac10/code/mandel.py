import numpy as np

# Función de cálculo del fractal en Python
def mandelPy(xmin, ymin, xmax, ymax, maxiter, xres, yres, A):
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
    A2D=vect.astype(np.ubyte).reshape(yres,xres) # row-major por defecto
    im=Image.fromarray(A2D)
    im.save(output)
