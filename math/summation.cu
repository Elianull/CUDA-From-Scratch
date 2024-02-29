#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "summation.cuh"

#define BLOCK_SIZE 256 // Number of threads per block

__global__ void summation(float *input, float *output, int len) {
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
        output[blockIdx.x] = shared[0];
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
        //input[i] = std::strtol(argv[i + 1], nullptr, 10);
        input[i] = std::atof(argv[i+1]);
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Assuming N is not too large for a single block
    summation<<<1, BLOCK_SIZE>>>(d_input, d_output, N);

    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sum: " << result << std::endl;

    delete[] input;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
#endif
