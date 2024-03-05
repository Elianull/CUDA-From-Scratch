#ifndef MATMUL_CUH
#define MATMUL_CUH

#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define TILE_WIDTH 16

void matmul(const std::vector<std::vector<float>>& matrixA, const std::vector<std::vector<float>>& matrixB, std::vector<std::vector<float>>& matrixC, int ARows, int ACols, int BCols);

#endif
