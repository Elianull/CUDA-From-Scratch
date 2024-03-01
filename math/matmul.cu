#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>
#include "dotprod.cuh"

// Function to generate a matrix with random values
std::vector<std::vector<float>> generateRandomMatrix(int rows, int cols) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

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

    return 0;
}
#endif
