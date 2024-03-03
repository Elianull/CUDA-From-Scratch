#ifndef MATMUL_CUH
#define MATMUL_CUH

#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void matmul(float* A, float* B, float* C, int ARows, int ACols, int BCols);

#endif
