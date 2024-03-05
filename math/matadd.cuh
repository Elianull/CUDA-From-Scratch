#ifndef MATADD_CUH
#define MATADD_CUH

#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define TILE_WIDTH 16

void matAdd(const std::vector<std::vector<float>>& matrixA, const std::vector<std::vector<float>>& matrixB, std::vector<std::vector<float>>& matrixC, int width, int height);

#endif
