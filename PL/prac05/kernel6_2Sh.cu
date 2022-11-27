#include "Prototipos.h"

int main(int argc, char *argv[]) {

    double *Hx, *Hy, *Hs;
    int ndev;
    float milliseconds;

    cudaEvent_t start, stop;

    int n = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int veces = atoi(argv[3]);
    int seed = atoi(argv[4]);

    CHECKNULL(Hx = (double *)malloc(n * sizeof(double)));
    CHECKNULL(Hy = (double *)malloc(n * sizeof(double)));
    CHECKNULL(Hs = (double *)malloc(n * sizeof(double)));

    Genera(Hx, n, seed);
    Genera(Hy, n, seed+18);

    cudaError_t error = cudaGetDeviceCount(&ndev);
    if (ndev == 0 || error != 0) {
        printf("No hay dispositivos CUDA disponibles");
        return EXIT_FAILURE;
    }

    CUDAERR(cudaMallocManaged((void**)&Hx, n * sizeof(double)));
    CUDAERR(cudaMallocManaged((void**)&Hy, n * sizeof(double)));
    CUDAERR(cudaMallocManaged((void**)&Hs, n * sizeof(double)));

    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // cpu
    for (int i = 0; i < veces; i++) {
        for (int j = 0; j < n; j++) {
            Hs[j] = Hx[j] + Hy[j];
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CPU: %f ms\n", milliseconds);

    cudaEventRecord(start);
    for (int i = 0; i < veces; i++) {
        kernel6_1<<<numBlocks, threadsPerBlock>>>(Hx, Hy, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA: %f ms\n", milliseconds);

    printf("\nError: %2.7E\n", Error(n, Hs, Hx));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(Hx);
    cudaFree(Hy);
    cudaFree(Hs);
}