#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256 // Number of threads per block

__global__ void summation(int *input, int *output, int len) {
    __shared__ int shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (i < len) ? input[i] : 1;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] *= shared[tid + s];
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
    int *input = new int[N];
    for (int i = 0; i < N; ++i) {
        input[i] = std::strtol(argv[i + 1], nullptr, 10);
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Assuming N is not too large for a single block
    summation<<<1, BLOCK_SIZE>>>(d_input, d_output, N);

    int result = 1;
    cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sum: " << result << std::endl;

    delete[] input;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
#endif
