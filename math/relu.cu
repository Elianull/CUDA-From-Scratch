#include <iostream>
#include <cstdlib>

__global__ void reluKernel(float* input, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = max(0.0f, input[index]);
    }
}

void relu(float* input, float* output, int size) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    reluKernel<<<numBlocks, blockSize>>>(d_input, d_output, size);

    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

#ifdef COMPILE_MAIN
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <size>\n";
        return 1;
    }

    int size = std::atoi(argv[1]);
    float* input = new float[size];
    float* output = new float[size];

    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f; // Random float between -10 and 10
    }

    relu(input, output, size);

    for (int i = 0; i < size; ++i) {
        std::cout << "ReLU(" << input[i] << ") = " << output[i] << std::endl;
    }

    delete[] input;
    delete[] output;
    return 0;
}
#endif
