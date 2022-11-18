#include "Prototipos.h"

int main(int argc, char *argv[]) {

    double *Hx, *Hy, *HA, *Hs;
    int ndev;
    double *Dx, *Dy, *DA;
    float milliseconds;

    cudaEvent_t start, stop;

    int n = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int veces = atoi(argv[3]);
    int seed = atoi(argv[4]);

    CHECKNULL(Hx = (double *)malloc(n * sizeof(double)));
    CHECKNULL(Hy = (double *)malloc(n * sizeof(double)));
    CHECKNULL(Hs = (double *)malloc(n * sizeof(double)));

    CHECKNULL(HA = (double *)malloc(n * n * sizeof(double)));

    Genera(Hx, n, seed);
    Genera(Hy, n, seed+18);
    Genera(HA, n*n, seed+36);

    cudaError_t error = cudaGetDeviceCount(&ndev);
    if (ndev == 0 || error != 0) {
        printf("No hay dispositivos CUDA disponibles");
        return EXIT_FAILURE;
    }

    CUDAERR(cudaMalloc((void **)&Dx, n * sizeof(double)));
    CUDAERR(cudaMalloc((void **)&Dy, n * sizeof(double)));
    CUDAERR(cudaMalloc((void **)&DA, n * n * sizeof(double)));

    CUDAERR(cudaMemcpy(Dx, Hx, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDAERR(cudaMemcpy(Dy, Hy, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDAERR(cudaMemcpy(DA, HA, n * n * sizeof(double), cudaMemcpyHostToDevice));

    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // cpu
    for (int i = 0; i < veces; i++) {
        for (int j = 0; j < n; j++) {
            Hs[j] = 0;
            for (int k = 0; k < n; k++) {
                Hs[j] += HA[j * n + k] * Hx[k];
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CPU: %f ms\n", milliseconds);

    cudaEventRecord(start);
    for (int i = 0; i < veces; i++) {
        kernel6_2<<<numBlocks, threadsPerBlock>>>(Dx, Dy, DA, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA: %f ms\n", milliseconds);

    CUDAERR(cudaMemcpy(Hx, Dy, n * sizeof(double), cudaMemcpyDeviceToHost));
    printf("\nError: %2.7E\n", Error(n, Hs, Hx));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(Dx);
    cudaFree(Dy);
    cudaFree(DA);
    free(Hx);
    free(Hy);
    free(HA);
    free(Hs);
}