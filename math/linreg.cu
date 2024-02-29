#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "linreg.cuh"
#include "mean.cuh"
#include "summation.cuh"

__global__ void prepareXYandX2(float *x_values, float *y_values, float *xy_values, float *x2_values, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        xy_values[idx] = x_values[idx] * y_values[idx];
        x2_values[idx] = x_values[idx] * x_values[idx];
    }
}

__global__ void linearRegression(float *d_sums, float *d_m, float *d_b, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum_x = d_sums[0];
        float sum_y = d_sums[1];
        float sum_xy = d_sums[2];
        float sum_x2 = d_sums[3];

        float m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x * sum_x);
        
        float b = (sum_y - m * sum_x) / N;

        *d_m = m;
        *d_b = b;
    }
}

void computeLinearRegression(float *d_x_values, float *d_y_values, int N, float *d_m, float *d_b) {
    float *d_sums, *d_xy_values, *d_x2_values;
    cudaMalloc(&d_sums, 4 * sizeof(float));
    cudaMalloc(&d_xy_values, N * sizeof(float));
    cudaMalloc(&d_x2_values, N * sizeof(float));

    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    prepareXYandX2<<<blocksPerGrid, BLOCK_SIZE>>>(d_x_values, d_y_values, d_xy_values, d_x2_values, N);

    cudaDeviceSynchronize();

    summation<<<blocksPerGrid, BLOCK_SIZE>>>(d_x_values, &d_sums[0], N); // sum_x
    summation<<<blocksPerGrid, BLOCK_SIZE>>>(d_y_values, &d_sums[1], N); // sum_y
    summation<<<blocksPerGrid, BLOCK_SIZE>>>(d_xy_values, &d_sums[2], N); // sum_xy
    summation<<<blocksPerGrid, BLOCK_SIZE>>>(d_x2_values, &d_sums[3], N); // sum_x2

    cudaDeviceSynchronize();

    linearRegression<<<1, 1>>>(d_sums, d_m, d_b, N);

    // Cleanup
    cudaFree(d_sums);
    cudaFree(d_xy_values);
    cudaFree(d_x2_values);
}

#ifdef COMPILE_MAIN
int main(int argc, char *argv[]) {
    if (argc <= 1 || argc % 2 != 1) {
        std::cout << "Usage: " << argv[0] << " <x1> <y1> <x2> <y2> ... <xN> <yN>\n";
        return 1;
    }

    int size = (argc - 1) / 2;
    size_t bytes = size * sizeof(float);

    float *x_values = (float*)malloc(bytes);
    float *y_values = (float*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        x_values[i] = std::atof(argv[1 + i]);
        y_values[i] = std::atof(argv[1 + size + i]);
    }

    float *d_x_values, *d_y_values;
    float *d_m_value, *d_b_value;

    cudaMalloc((void**)&d_x_values, bytes);
    cudaMalloc((void**)&d_y_values, bytes);
    cudaMalloc((void**)&d_m_value, sizeof(float));
    cudaMalloc((void**)&d_b_value, sizeof(float));

    cudaMemcpy(d_x_values, x_values, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_values, y_values, bytes, cudaMemcpyHostToDevice);

    computeLinearRegression(d_x_values, d_y_values, size, d_m_value, d_b_value);

    float m_value, b_value;
    cudaMemcpy(&m_value, d_m_value, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b_value, d_b_value, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Slope (m): " << m_value << ", Intercept (b): " << b_value << std::endl;

    cudaFree(d_x_values);
    cudaFree(d_y_values);
    cudaFree(d_m_value);
    cudaFree(d_b_value);

    free(x_values);
    free(y_values);

    return 0;
}
#endif
