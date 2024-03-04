#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>
#include "dotprod.cuh"

#define TILE_WIDTH 16

// Function to generate a matrix with random values
std::vector<std::vector<float>> generateRandomMatrix(int rows, int cols) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}

// Function to print a matrix
void printMatrix(const std::vector<std::vector<float>>& matrix) {
    for (const auto &row : matrix) {
        for (auto elem : row) {
            std::cout << std::fixed << std::setprecision(4) << elem << " ";
        }
        std::cout << std::endl;
    }
}

float* flattenMatrix(const std::vector<std::vector<float>>& matrix, int rows, int cols) {
    float* flat = new float[rows * cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
    return flat;
}


// __global__ void matmul(float* A, float* B, float* C, int ARows, int ACols, int BCols) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if(row < ARows && col < BCols) {
//         float sum = 0.0;
//         for (int k = 0; k < ACols; ++k) {
//             sum += A[row * ACols + k] * B[k * BCols + col];
//         }
//         C[row * BCols + col] = sum;
//     }
// }

__global__ void matmul(float* A, float* B, float* C, int ARows, int ACols, int BCols) {
    __shared__ float Asub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bsub[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0;
    // Loop over sub-matrices of A and B that are required to compute the C element
    for (int k = 0; k < (ACols + TILE_WIDTH - 1) / TILE_WIDTH; k++) {
        // Load sub-matrices into shared memory
        if (k*TILE_WIDTH + threadIdx.x < ACols && row < ARows)
            Asub[threadIdx.y][threadIdx.x] = A[row*ACols + k*TILE_WIDTH + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0;

        if (k*TILE_WIDTH + threadIdx.y < BCols && col < BCols)
            Bsub[threadIdx.y][threadIdx.x] = B[(k*TILE_WIDTH + threadIdx.y)*BCols + col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int n = 0; n < TILE_WIDTH; ++n)
            sum += Asub[threadIdx.y][n] * Bsub[n][threadIdx.x];

        __syncthreads();
    }
    
    if (row < ARows && col < BCols)
        C[row*BCols + col] = sum;
}


#ifdef COMPILE_MAIN
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <m> <n> <p>\n";
        return 1;
    }

    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int p = std::atoi(argv[3]);

    auto matrixA = generateRandomMatrix(m, n);
    auto matrixB = generateRandomMatrix(n, p);

    std::cout << "Matrix A (" << m << "x" << n << "):\n";
    printMatrix(matrixA);
    std::cout << "\nMatrix B (" << n << "x" << p << "):\n";
    printMatrix(matrixB);

        // Flatten matrices
    float* A_flat = flattenMatrix(matrixA, m, n);
    float* B_flat = flattenMatrix(matrixB, n, p);
    float* C_flat = new float[m * p];

    // CUDA memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * p * sizeof(float));
    cudaMalloc(&d_C, m * p * sizeof(float));

    cudaMemcpy(d_A, A_flat, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat, n * p * sizeof(float), cudaMemcpyHostToDevice);

    //const int TILE_WIDTH = 16;
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);

    cudaMemcpy(C_flat, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<std::vector<float>> matrixC(m, std::vector<float>(p));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            matrixC[i][j] = C_flat[i * p + j];
        }
    }
    std::cout << "\nResult Matrix C (" << m << "x" << p << "):\n";
    printMatrix(matrixC);

    delete[] A_flat;
    delete[] B_flat;
    delete[] C_flat;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
#endif
