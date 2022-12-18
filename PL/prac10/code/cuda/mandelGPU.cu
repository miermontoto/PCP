#include "PrototiposGPU.h"

__global__ void kernelMandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A)
{
	double dx = (xmax - xmin) / xres;
	double dy = (ymax - ymin) / yres;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	double x, y, u, v, u2, v2;
	int k;


	x = xmin + i * dx;
	y = ymin + j * dy;

	u = 0.0;
	v = 0.0;
	u2 = u * u;
	v2 = v * v;
	k = 1;

	while (u2 + v2 < 4.0 && k < maxiter) {
		v = 2.0 * u * v + y;
		u = u2 - v2 + x;
		u2 = u * u;
		v2 = v * v;
		k++;
	}

	A[i + j * xres] = k >= maxiter ? 0 : k;

}


__global__ void kernelBinariza(int xres, int yres, double* A, double med) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < xres && j < yres) {
		A[i + j * xres] = A[i + j * xres] > med ? 255 : 0;
	}
}

// Kernel auxiliar para calcular el promedio.
// Sirve de barrera de sincronización para que todos los bloques terminen de calcluar su suma.
__global__ void kernelPromedio_sumBlocksValue(double* b_sum, double* sum, int numBlocks) {
	extern __shared__ double cache[];
	int blocksPerThread = numBlocks / blockDim.x;
	int tid = threadIdx.x;

	double temp = 0;
	for (int i = 0; i < blocksPerThread; i++) {
		temp += b_sum[blocksPerThread * tid + i];
	}

	cache[tid] = temp;
	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0) {
		if (tid < i) {
			cache[tid] += cache[tid + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (tid == 0) {
		*sum = *cache;
	}
}

__global__ void kernelPromedio_getBlocksValue(int xres, int yres, double* A, double* b_sum) {
	extern __shared__ double cache[]; // Almacena la suma de cada espacio bloque / hilos.
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	// Cada hilo calcula su suma.
	double temp = 0;
	while (tid < xres * yres) {
		temp += A[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = temp; // Se guarda la suma de cada hilo en memoria compartida.

	__syncthreads();

	// Se reduce y se obtiene la suma de cada bloque.
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) { // Un solo hilo:
		b_sum[blockIdx.x] = cache[0]; // almacena la suma total de cada bloque.
	}
}

// --- Funciones en C --- //
// Función estandar de mandel. Utiliza memoria convencional y funciona en 2D.
extern "C" void mandelGPU_normal(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk) {
	int size = xres * yres * sizeof(double);
	double* d_A;

	CUDAERR(cudaMalloc((void**) &d_A, size));

	dim3 dimBlock(ThpBlk, ThpBlk);
	dim3 dimGrid((xres + dimBlock.x - 1) / dimBlock.x, (yres + dimBlock.y - 1) / dimBlock.y);

	kernelMandel<<<dimGrid, dimBlock>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, d_A);
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost));
	cudaFree(d_A);
}

extern "C" int mandel_iter(double x, double y, int maxiter) {

      double u = 0.0, v = 0.0;
      double u2 = 0.0, v2 = 0.0;

      int k = 1;
      while (k < maxiter && u2 + v2 < 4.0) {
         v = 2.0 * u * v + y;
         u = u2 - v2 + x;
         u2 = u * u;
         v2 = v * v;
         k++;
      }

      return k == maxiter ? 0 : k;
}

// Función similar a mandelGPU, pero la CPU realiza el 10% del cálculo y el 90% lo realiza la GPU.
extern "C" void mandelGPU_heter(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk) {
	int size = xres * yres * sizeof(double);
	double* d_A;

	CUDAERR(cudaMallocManaged((void**) &d_A, size));

	dim3 dimBlock(ThpBlk, ThpBlk);
	dim3 dimGrid((xres * 0.9 + dimBlock.x - 1) / dimBlock.x, (yres + dimBlock.y - 1) / dimBlock.y);

	kernelMandel<<<dimGrid, dimBlock>>>(xmin, ymin, xmax, ymax, maxiter, xres * 0.9, yres, d_A);

	// Se realiza el 10% restante de cálculo en la CPU.
	for (int i = xres * 0.9; i < xres; i++) {
		for (int j = 0; j < yres; j++) {
			d_A[i * yres + j] = mandel_iter(xmin + (xmax - xmin) * i / xres, ymin + (ymax - ymin) * j / yres, maxiter);
		}
	}

	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost));
	cudaFree(d_A);
}

// Función mandel con Pinned Memory.
extern "C" void mandelGPU_pinned(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk) {
	int size = xres * yres * sizeof(double);
	double* d_A, * h_A;

	CUDAERR(cudaHostAlloc((void**) &h_A, size, cudaHostAllocMapped));
	CUDAERR(cudaHostGetDevicePointer((void**) &d_A, (void*) h_A, 0));
	CUDAERR(cudaMemcpy(h_A, A, size, cudaMemcpyHostToHost));

	dim3 dimBlock(ThpBlk, ThpBlk);
	dim3 dimGrid((xres + dimBlock.x - 1) / dimBlock.x, (yres + dimBlock.y - 1) / dimBlock.y);

	kernelMandel<<<dimGrid, dimBlock>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, d_A);
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, h_A, size, cudaMemcpyHostToHost));
	CUDAERR(cudaFreeHost(h_A));
	// NO se libera d_A porque es un puntero a la memoria compartida. (<- mucho sufrimiento.)
}

// Función mandel con Unified Memory.
extern "C" void mandelGPU_unified(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk) {
	int size = xres * yres * sizeof(double);
	double* d_A;

	CUDAERR(cudaMallocManaged((void**) &d_A, size)); // no se especifica el flag cudaMemAttachGlobal (valor por defecto)
	CUDAERR(cudaMemcpy(d_A, A, size, cudaMemcpyDefault)); // no sé muy bien dónde está d_A → dejo que lo infiera.

	dim3 dimBlock(ThpBlk, ThpBlk);
	dim3 dimGrid((xres + dimBlock.x - 1) / dimBlock.x, (yres + dimBlock.y - 1) / dimBlock.y);

	kernelMandel<<<dimGrid, dimBlock>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, d_A);
	CHECKLASTERR();

	CUDAERR(cudaDeviceSynchronize());
	cudaMemcpy(A, d_A, size, cudaMemcpyDefault);
	cudaFree(d_A);
}

// -- Promedios

// Función que calcula el promedio utilizando la API de cublas.
extern "C" double promedioGPU_api(int xres, int yres, double* A, int ThpBlk) {
	int size = xres * yres;

	double sum;
	double* d_A;

	cublasHandle_t handle;
	if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS) {
		printf("Error al crear el handle de cublas.");
		exit(EXIT_FAILURE);
	}

	CUDAERR(cudaMalloc((void**) &d_A, size * sizeof(double)));
	CUDAERR(cudaMemcpy(d_A, A, size * sizeof(double), cudaMemcpyHostToDevice));
	cublasDasum_v2(handle, size, d_A, 1, &sum);
	CHECKLASTERR();

	cublasDestroy_v2(handle); // Se utiliza "v2" pero en realidad la función original expande a la versión 2 por defecto
	cudaFree(d_A);
	return sum / size;
}

// Función promedio con Shared Memory.
// Inspirado en el Problema X de las PAs.
extern "C" double promedioGPU_shared(int xres, int yres, double* A, int ThpBlk) {
	int size = xres * yres * sizeof(double);
	double* d_A;
	double h_sum, *d_sum, *b_sum;

	int dimGrid = ((xres * yres + ThpBlk - 1) / ThpBlk);

	CUDAERR(cudaMalloc((void**) &d_A, size));
	CUDAERR(cudaMalloc((void**) &d_sum, sizeof(double)));
	CUDAERR(cudaMalloc((void**) &b_sum, dimGrid * sizeof(double)));
	CUDAERR(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

	kernelPromedio_getBlocksValue<<<dimGrid, ThpBlk, ThpBlk * sizeof(double)>>>(xres, yres, d_A, b_sum);
	kernelPromedio_sumBlocksValue<<<1, 1024, 1024 * sizeof(double)>>>(b_sum, d_sum, dimGrid);
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(d_A);
	cudaFree(d_sum);
	cudaFree(b_sum);
	return h_sum / (xres * yres);
}

// -- Binarización

extern "C" void binarizaGPU(int xres, int yres, double* A, double med, int ThpBlk) {
	int size = xres * yres * sizeof(double);
	double* d_A;

	CUDAERR(cudaMalloc((void**) &d_A, size));
	CUDAERR(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

	dim3 dimBlock(ThpBlk, ThpBlk);
	dim3 dimGrid((xres + dimBlock.x - 1) / dimBlock.x, (yres + dimBlock.y - 1) / dimBlock.y);

	kernelBinariza<<<dimGrid, dimBlock>>>(xres, yres, d_A, med);
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost));
	cudaFree(d_A);
}
