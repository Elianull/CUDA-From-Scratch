#ifndef MEAN_CUH
#define MEAN_CUH

#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void mean(float *input, float *output, int len);

#endif
