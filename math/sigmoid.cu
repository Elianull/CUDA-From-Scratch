#include <iostream>
#include <cmath>
#include <cstdlib> // For atof

__global__ void sigmoidKernel(float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
}

void sigmoid(float* input, float* output, int N) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    sigmoidKernel<<<numBlocks, blockSize>>>(d_input, d_output, N);

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

#ifdef COMPILE_MAIN
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N>\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    float* input = new float[N];
    float* output = new float[N];

    for (int i = 0; i < N; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random values between -1 and 1
    }

    sigmoid(input, output, N);

    for (int i = 0; i < N; ++i) {
        std::cout << "Sigmoid(" << input[i] << ") = " << output[i] << std::endl;
    }

    delete[] input;
    delete[] output;
    return 0;
}
#endif
