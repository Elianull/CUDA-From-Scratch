#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "mean.cuh"

#define BLOCK_SIZE 256 // Number of threads per block

#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
}

__global__ void mean(float *input, float *output, int len) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (i < len) ? input[i] : 0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = shared[0] / len;
    }
}

#ifdef COMPILE_MAIN
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <element1> <element2> ... <elementN>" << std::endl;
        return 1;
    }

    int N = argc - 1;
    float *input = new float[N];
    for (int i = 0; i < N; ++i) {
        input[i] = std::atof(argv[i + 1]);
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Assuming N is not too large for a single block
    mean<<<1, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaCheckError();

    float result;
    cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Mean: " << result << std::endl;

    delete[] input;

    return 0;
}
#endif
