#ifndef MEAN_CUH
#define MEAN_CUH

#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void mean(int *input, int *output, int len);

#endif
