#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void linearRegression(int *x_values, int *y_values, float *m_value, float *b_value, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {

    }
}

int main(int argc, char *argv[]) {
    if (argc <= 1 || argc % 2 != 1) {
        printf("Usage: linreg.cu <elem1> <elem2> ... <elemN>\n");
        printf("The number of elements must be even.\n");
        return 1;
    }

    int size = (argc - 1) / 2;
    size_t bytes = size * sizeof(int);

    int *x_values = (int*)malloc(bytes);
    int *y_values = (int*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        x_values[i] = atoi(argv[1 + i]);
        y_values[i] = atoi(argv[1 + size + i]);
    }

    int *d_x_values, *d_y_values;
    float *d_m_value, *d_b_value;

    cudaMalloc(&d_x_values, bytes);
    cudaMalloc(&d_y_values, bytes);
    cudaMalloc(&d_m_value, sizeof(float));
    cudaMalloc(&d_b_value, sizeof(float));


    cudaMemcpy(d_x_values, x_values, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_values, y_values, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    linearRegression<<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, d_m_value, d_b_value, size);

    float m_value, b_value;
    cudaMemcpy(&m_value, d_m_value, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b_value, d_b_value, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x_values);
    cudaFree(d_y_values);
    cudaFree(d_m_value);
    cudaFree(d_b_value);

    free(x_values);
    free(y_values);

    return 0;
}