#ifndef SUMMATION_CUH
#define SUMMATION_CUH

#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void summation(float *input, float *output, int len);

#endif
