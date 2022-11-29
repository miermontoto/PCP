#include "Prototipos.h"
#include "Kernels.h"

void kernel6_1(int, int, int, cudaEvent_t, cudaEvent_t, bool);
void kernel6_2(int, int, int, cudaEvent_t, cudaEvent_t);

int main(int argc, char *argv[]) {

    int ndev;
    float milliseconds;
    cudaEvent_t start, stop;

    int n = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int repetitions = atoi(argv[3]);

    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t rev = cudaSetDevice(cudaGetDeviceCount(&ndev));
    if (ndev == 0 || rev != cudaSuccess) {
        printf("No CUDA devices found");
        return EXIT_FAILURE;
    }

    kernel6_1(n, threadsPerBlock, repetitions, start, stop, false);


}

void kernel6_1(int size, int tpb, int rept, cudaEvent_t start, cudaEvent_t stop, bool shared) {
    double *Hx, *Hy, *Hs;
    double *Dx, *Dy, *Dv;

    unsigned int seed;
    srand(seed);

    CHECKNULL(Hx = (double *)malloc(size * sizeof(double)));
    CHECKNULL(Hy = (double *)malloc(size * sizeof(double)));
    CHECKNULL(Hs = (double *)malloc(size * sizeof(double)));

    Genera(Hx, size, seed);
    Genera(Hy, size, seed+18);

    CUDAERR(cudaMalloc((void **)&Dx, size * sizeof(double)));
    CUDAERR(cudaMalloc((void **)&Dy, size * sizeof(double)));

    CUDAERR(cudaMemcpy(Dx, Hx, size * sizeof(double), cudaMemcpyHostToDevice));
    CUDAERR(cudaMemcpy(Dy, Hy, size * sizeof(double), cudaMemcpyHostToDevice));

    int numBlocks = (size + tpb - 1) / tpb;

    // CPU
    for(int i = 0; i < rept; i++) {
        cudaEventRecord(start);
        for(int j = 0; j < size; j++) {
            Hs[j] = Hx[j] + Hy[j];
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("CPU: %f ms", milliseconds);
    }

    // CUDA
    for(int i = 0; i < rept; i++) {
        cudaEventRecord(start);
        if(shared) {
            kernel6_1Sh<<<numBlocks, tpb>>>(size, Dx, Dy, Dv);
        } else {
            kernel6_1<<<numBlocks, tpb>>>(size, Dx, Dy, Dv);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("CUDA: %f ms", milliseconds);
    }

}

