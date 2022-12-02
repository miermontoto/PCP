
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

__global__ void kernelBinariza(int xres, int yres, double* A, double med){

}

extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxk, int xres, int yres, double* A, int ThpBlk) {

	int size = xres * yres * sizeof(double);
	double* d_A;

	CUDAERR(cudaMalloc((void**) &d_A, size));

	dim3 dimBlock(ThpBlk, ThpBlk);
	dim3 dimGrid((xres + dimBlock.x - 1) / dimBlock.x, (yres + dimBlock.y - 1) / dimBlock.y);

	kernelMandel<<<dimGrid, dimBlock>>>(xmin, ymin, xmax, ymax, maxk, xres, yres, d_A);
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost));
	cudaFree(d_A);

}

extern "C" double promedioGPU(int xres, int yres, double* A, int ThpBlk) {
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

	cublasDestroy_v2(handle);
	cudaFree(d_A);
	return sum / size;
}

extern "C" void binarizaGPU(int xres, int yres, double* A, double med, int ThpBlk) {

}
