#ifndef CROSSENTROPYLOSS_CUH
#define CROSSENTROPYLOSSD_CUH

#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define TILE_WIDTH 16

float crossEntropyLoss(float* predictions, float* targets, int numClasses, int totalSize, int batchSize);

#endif
