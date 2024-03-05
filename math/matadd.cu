#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>

#define TILE_WIDTH 16

// Function to generate a matrix with random values
inline std::vector<std::vector<float>> generateRandomMatrix(int rows, int cols) {
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
inline void printMatrix(const std::vector<std::vector<float>>& matrix) {
    for (const auto &row : matrix) {
        for (auto elem : row) {
            std::cout << std::fixed << std::setprecision(4) << elem << " ";
        }
        std::cout << std::endl;
    }
}

inline float* flattenMatrix(const std::vector<std::vector<float>>& matrix, int rows, int cols) {
    float* flat = new float[rows * cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
    return flat;
}

__global__ void matrixAddKernel(float* A, float* B, float* C, int width, int height) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column < width && row < height) {
        int index = row * width + column;
        C[index] = A[index] + B[index];
    }
}

void matAdd(const std::vector<std::vector<float>>& matrixA,
            const std::vector<std::vector<float>>& matrixB,
            std::vector<std::vector<float>>& matrixC,
            int width, int height) {
    
    // Flatten matrices A and B
    float* A_flat = new float[width * height];
    float* B_flat = new float[width * height];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            A_flat[i * width + j] = matrixA[i][j];
            B_flat[i * width + j] = matrixB[i][j];
        }
    }
    
    float* C_flat = new float[width * height];

    float *d_A, *d_B, *d_C;
    size_t size = width * height * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A_flat, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width, height);

    cudaMemcpy(C_flat, d_C, size, cudaMemcpyDeviceToHost);

    // Reconstruct matrix C from C_flat
    matrixC.resize(height);
    for (int i = 0; i < height; ++i) {
        matrixC[i].resize(width);
        for (int j = 0; j < width; ++j) {
            matrixC[i][j] = C_flat[i * width + j];
        }
    }

    delete[] A_flat;
    delete[] B_flat;
    delete[] C_flat;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#ifdef COMPILE_MAIN
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <width> <height>\n";
        return 1;
    }

    int width = std::atoi(argv[1]);
    int height = std::atoi(argv[2]);

    // Generate and print matrices A and B
    auto matrixA = generateRandomMatrix(height, width); // Adjusting to height, width for clarity
    auto matrixB = generateRandomMatrix(height, width);

    std::cout << "Matrix A (" << height << "x" << width << "):\n";
    printMatrix(matrixA);
    std::cout << "\nMatrix B (" << height << "x" << width << "):\n";
    printMatrix(matrixB);

    std::vector<std::vector<float>> matrixC;

    matAdd(matrixA, matrixB, matrixC, width, height);

    std::cout << "\nResult Matrix C (" << height << "x" << width << "):\n";
    printMatrix(matrixC);

    return 0;
}
#endif
