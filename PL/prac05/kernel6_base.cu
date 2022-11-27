#include "Prototipos.h"
#include "Kernels.h"

int main(int argc, char *argv[]) {

    int ndev;
    float milliseconds;

    int n = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int repetitions = atoi(argv[3]);
    int seed = atoi(argv[4]);

    cudaError_t rev = cudaSetDevice(cudaGetDeviceCount(&ndev));
    if (ndev == 0 || rev != cudaSuccess) {
        printf("No CUDA devices found");
        return EXIT_FAILURE;
    }
}

