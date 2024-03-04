#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "mean.cuh"
#include "summation.cuh"

#define BLOCK_SIZE 256 // Number of threads per block

__global__ void distanceToMeanSquared(float *d_values, float mean, float *d_distance, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        d_distance[index] = abs(d_values[index] - mean);
        d_distance[index] = d_distance[index]*d_distance[index];
    }
}

void computeStandardDeviation(float *d_values, int N, float &h_stdDev) {
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float h_mean;
    float *d_mean;
    float *d_distance;
    float *d_sum;
    float h_sum;

    cudaMalloc(&d_mean, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_distance, N*sizeof(float));

    mean<<<1, BLOCK_SIZE>>>(d_values, d_mean, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
    //std::cout << "Mean: " << h_mean << std::endl;

    distanceToMeanSquared<<<gridSize, BLOCK_SIZE>>>(d_values, h_mean, d_distance, N);
    cudaDeviceSynchronize();

    summation<<<1, BLOCK_SIZE>>>(d_distance, d_sum, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    //std::cout << "Sum of Squared Differences: " << h_sum << std::endl;

    float variance = h_sum / N;
    h_stdDev = sqrt(variance);

    // Cleanup
    cudaFree(d_mean);
    cudaFree(d_sum);
    cudaFree(d_distance);
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

    float h_stdDev;
    computeStandardDeviation(d_input, N, h_stdDev);

    std::cout << "Standard Deviation: " << h_stdDev << std::endl;

    delete[] input;

    return 0;
}
#endif
