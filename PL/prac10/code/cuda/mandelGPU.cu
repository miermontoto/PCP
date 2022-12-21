#include "PrototiposGPU.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

__global__ void kernelMandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= xres || j >= yres) return;

	double dx = (xmax - xmin) / xres;
	double dy = (ymax - ymin) / yres;

	double x = xmin + i * dx;
	double y = ymin + j * dy;

	double u = 0.0;
	double v = 0.0;
	double u2 = u * u;
	double v2 = v * v;

	int k = 1;
	while (u2 + v2 < 4.0 && k < maxiter) {
		v = 2.0 * u * v + y;
		u = u2 - v2 + x;
		u2 = u * u;
		v2 = v * v;
		k++;
	}

	A[i + j * xres] = k >= maxiter ? 0 : k;
}

__global__ void kernelMandel_1D(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= xres) return;

	double dx = (xmax - xmin) / xres;
	double dy = (ymax - ymin) / yres;

	int j;
	for(j = 0; j < yres; j++) {
		double x = xmin + tid * dx;
		double y = ymin + j * dy;

		double u = 0.0;
		double v = 0.0;
		double u2 = u * u;
		double v2 = v * v;

		int k = 1;
		while (u2 + v2 < 4.0 && k < maxiter) {
			v = 2.0 * u * v + y;
			u = u2 - v2 + x;
			u2 = u * u;
			v2 = v * v;
			k++;
		}

		A[tid + j * xres] = k >= maxiter ? 0 : k;
	}
}

__global__ void kernelBinariza(int xres, int yres, double* A, double med) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < xres && j < yres) {
		A[i + j * xres] = A[i + j * xres] > med ? 255 : 0;
	}
}

// Kernel auxiliar para calcular el promedio.
// Sirve de barrera de sincronización para que todos los bloques terminen de calcuar su suma.
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

__global__ void kernelPromedio_getBlocksValue_ns(int xres, int yres, double* A, double* b_sum, double* cache) {
	int cacheIndex = threadIdx.x;
	int source = blockIdx.x * blockDim.x;
	int tid = cacheIndex + source;

	// Cada hilo calcula su suma.
	double temp = 0;
	while (tid < xres * yres) {
		temp += A[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[source + cacheIndex] = temp;

	__syncthreads();

	// Se reduce y se obtiene la suma de cada bloque.
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[source + cacheIndex] += cache[source + cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) { // Un solo hilo:
		b_sum[blockIdx.x] = cache[source]; // almacena la suma total de cada bloque.
	}
}

__global__ void kernelPromedio_atomic(int xres, int yres, double* A, unsigned long long int* sum) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < xres * yres) {
		atomicAdd(sum, (int) A[tid]);
	}
}

__global__ void kernelPromedio_sumBlocksValue_ns(double* b_sum, double* sum, int numBlocks, double* cache) {
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

int mandel(double x, double y, int maxiter) {

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

// Función similar a mandelGPU, pero la CPU realiza parte del cálculo de manera simultánea.
extern "C" void mandelGPU_heter(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk) {
	float percentage = 0.7;
	int basic_size = xres * yres;
	int size = basic_size * sizeof(double);
	double dx = (xmax - xmin) / xres;
    double dy = (ymax - ymin) / yres;
	double* d_A;

	CUDAERR(cudaMallocManaged((void**) &d_A, size));

	dim3 dimBlock(ThpBlk, ThpBlk);
    dim3 dimGrid((xres + dimBlock.x - 1) / dimBlock.x, (yres * percentage + dimBlock.y - 1) / dimBlock.y);

	kernelMandel<<<dimGrid, dimBlock>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, d_A);
	CHECKLASTERR();

	#pragma omp parallel for schedule(guided)
	for(int i = percentage * basic_size; i < basic_size; i++) {
		d_A[i] = mandel(xmin + (i % xres) * dx, ymin + (i / xres) * dy, maxiter);
	}

	CUDAERR(cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost));
	cudaFree(d_A);
}

extern "C" void mandelGPU_1D(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk) {
	int size = xres * yres * sizeof(double);
	double* d_A;

	CUDAERR(cudaMalloc((void**) &d_A, size));
	int numBlocks = (xres * yres + ThpBlk - 1) / ThpBlk;

	kernelMandel_1D<<<numBlocks, ThpBlk>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, d_A);
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
	int basic_size = xres * yres;
	int size = basic_size * sizeof(double);

	double sum;
	double* d_A;

	cublasHandle_t handle;
	if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS) {
		printf("Error al crear el handle de cublas.");
		exit(EXIT_FAILURE);
	}

	CUDAERR(cudaMalloc((void**) &d_A, size));
	CUDAERR(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
	cublasDasum_v2(handle, basic_size, d_A, 1, &sum);
	CHECKLASTERR();

	cublasDestroy_v2(handle); // Se utiliza "v2" pero en realidad la función original expande a la versión 2 por defecto
	cudaFree(d_A);
	return sum / basic_size;
}

// Función promedio con Shared Memory.
// Inspirado en el Problema X de las PAs.
extern "C" double promedioGPU_shared(int xres, int yres, double* A, int ThpBlk) {
	int basic_size = xres * yres;
	int size = basic_size * sizeof(double);
	double* d_A;
	double h_sum, *d_sum, *b_sum;

	int dimGrid = (xres * yres + ThpBlk - 1) / ThpBlk;

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
	return h_sum / basic_size;
}

extern "C" double promedioGPU_param(int xres, int yres, double* A, int ThpBlk) {
	int basic_size = xres * yres;
	int size = basic_size * sizeof(double);
	double* d_A, *cache;
	double h_sum, *d_sum, *b_sum;

	int dimGrid = (xres * yres + ThpBlk - 1) / ThpBlk;

	CUDAERR(cudaMalloc((void**) &d_A, size));
	CUDAERR(cudaMalloc((void**) &d_sum, sizeof(double)));
	CUDAERR(cudaMalloc((void**) &b_sum, dimGrid * sizeof(double)));
	CUDAERR(cudaMalloc((void**) &cache, dimGrid * ThpBlk * sizeof(double)));
	CUDAERR(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

	kernelPromedio_getBlocksValue_ns<<<dimGrid, ThpBlk>>>(xres, yres, d_A, b_sum, cache);
	kernelPromedio_sumBlocksValue_ns<<<1, 1024>>>(b_sum, d_sum, dimGrid, cache); // se reutiliza la caché
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(d_A);
	cudaFree(d_sum);
	cudaFree(cache);

	return h_sum / basic_size;
}

extern "C" double promedioGPU_atomic(int xres, int yres, double* A, int ThpBlk) {
	int basic_size = xres * yres;
	int size = basic_size * sizeof(double);
	double* d_A;
	unsigned long long int* d_sum, h_sum; // Tiene que ser long long int para evitar overflow en tamaños grandes.

	CUDAERR(cudaMalloc((void**) &d_A, size));
	CUDAERR(cudaMalloc((void**) &d_sum, sizeof(long long int)));
	CUDAERR(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
	CUDAERR(cudaMemset(d_sum, 0, sizeof(long long int)));

	int dimGrid = (xres * yres + ThpBlk - 1) / ThpBlk;

	kernelPromedio_atomic<<<dimGrid, ThpBlk>>>(xres, yres, d_A, d_sum);
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(&h_sum, d_sum, sizeof(long long int), cudaMemcpyDeviceToHost));
	cudaFree(d_sum);
	cudaFree(d_A);
	return (double) h_sum / basic_size;
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
