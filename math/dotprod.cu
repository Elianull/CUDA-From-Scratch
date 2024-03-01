#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "summation.cuh"

__global__ void elementWiseMultiply(float *x, float *y, float *result, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        result[index] = x[index] * y[index];
    }
}


void computeDotProduct(float *d_x_values, float *d_y_values, int N, float *d_dot_value) {
    float *d_multiples;
    cudaMalloc(&d_multiples, N * sizeof(float));


    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    elementWiseMultiply<<<blocksPerGrid, BLOCK_SIZE>>>(d_x_values, d_y_values, d_multiples, N);

    cudaDeviceSynchronize();

    summation<<<blocksPerGrid, BLOCK_SIZE>>>(d_multiples, d_dot_value, N);

    cudaDeviceSynchronize();

    cudaFree(d_multiples);
}

#ifdef COMPILE_MAIN
int main(int argc, char *argv[]) {
    if (argc <= 1 || argc % 2 != 1) {
        std::cout << "Usage: " << argv[0] << " <x1> <x2> <x3> ... <y1> <y2> <y3> ...\n";
        return 1;
    }

    int size = (argc - 1) / 2;
    size_t bytes = size * sizeof(float);

    float *x_values = (float*)malloc(bytes);
    float *y_values = (float*)malloc(bytes);

    for (int i = 1; i <= size; i++) {
        x_values[i - 1] = std::atof(argv[i]);
        y_values[i - 1] = std::atof(argv[i + size]);
    }

    float *d_x_values, *d_y_values, *d_dot_value;

    cudaMalloc((void**)&d_x_values, bytes);
    cudaMalloc((void**)&d_y_values, bytes);
    cudaMalloc((void**)&d_dot_value, sizeof(float));

    cudaMemcpy(d_x_values, x_values, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_values, y_values, bytes, cudaMemcpyHostToDevice);

    computeDotProduct(d_x_values, d_y_values, size, d_dot_value);

    float dot_value;
    cudaMemcpy(&dot_value, d_dot_value, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Dot product: " << dot_value << std::endl;

    cudaFree(d_x_values);
    cudaFree(d_y_values);
    cudaFree(d_dot_value);

    free(x_values);
    free(y_values);

    return 0;
}
#endif
