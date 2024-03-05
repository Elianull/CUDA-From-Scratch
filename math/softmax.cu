#include <iostream>
#include <cmath>
#include <cstdlib>
#include <numeric>

__global__ void softmaxKernel(float* input, float* output, int size) {
    extern __shared__ float temp[];
    int index = threadIdx.x;

    if (index < size) {
        temp[index] = exp(input[index]);
    }
    __syncthreads();

    if (index == 0) {
        float sum = 0.0;
        for (int i = 0; i < size; ++i) {
            sum += temp[i];
        }
        for (int i = 0; i < size; ++i) {
            output[i] = temp[i] / sum;
        }
    }
}

void softmax(float* input, float* output, int size) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    softmaxKernel<<<1, size, size * sizeof(float)>>>(d_input, d_output, size);

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
        input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f; // Random float between 0 and 10
    }

    softmax(input, output, size);

    std::cout << "Softmax Transformation\n";
    std::cout << "Input -> Output:\n";
    for (int i = 0; i < size; ++i) {
        std::cout << input[i] << " -> " << output[i] << std::endl;
    }

    delete[] input;
    delete[] output;
    return 0;
}
#endif
