
#include "PrototiposGPU.h"

__global__ void kernelMandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A)
{
  
}



__global__ void kernelBinariza(int xres, int yres, double* A, double med){
  
}

extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk){
   
}

extern "C" double promedioGPU(int xres, int yres, double* A, int ThpBlk){
	
	return -1;
}

extern "C" void binarizaGPU(int xres, int yres, double* A, double med, int ThpBlk){
	
}
