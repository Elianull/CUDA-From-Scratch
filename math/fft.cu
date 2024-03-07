#include <iostream>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>

const float PI = 2*acos(0.0);

__global__ void fftBreakdown(cuFloatComplex* input, cuFloatComplex* output, int size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int numPairs = (size+1) / 2;
    
    if (bid * blockDim.x + tid < numPairs) {
        int idx = bid * blockDim.x + tid;
        output[idx] = input[idx];
        output[idx + numPairs] = input[idx + numPairs];
    }
}

std::vector<std::vector<cuFloatComplex>> fftBreakdownHost(std::vector<cuFloatComplex>& input) {
    int size = input.size();
    int numPairs = (size + 1) / 2;
    bool oddSize = (size % 2 != 0);
    std::vector<std::vector<cuFloatComplex>> output(numPairs, std::vector<cuFloatComplex>(2));

    // Pad the input with a zero element if the size is odd
    if (oddSize) {
        input.push_back(make_cuFloatComplex(0.0f, 0.0f));
        size++;
    }

    // Allocate device memory
    cuFloatComplex* d_input;
    cuFloatComplex* d_output;
    cudaMalloc(&d_input, size * sizeof(cuFloatComplex));
    cudaMalloc(&d_output, size * sizeof(cuFloatComplex));

    // Copy input data to device
    cudaMemcpy(d_input, input.data(), size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // Launch the fftBreakdown kernel
    int blockSize = 256;
    int numBlocks = (numPairs + blockSize - 1) / blockSize;
    fftBreakdown<<<numBlocks, blockSize>>>(d_input, d_output, size);
    cudaDeviceSynchronize();  // Wait for kernel to finish

    // Copy the pairs back to the host
    for (int i = 0; i < numPairs; i++) {
        cudaMemcpy(&output[i][0], &d_output[i], sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(&output[i][1], &d_output[i + numPairs], sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

std::vector<cuFloatComplex> computeFFT(std::vector<cuFloatComplex>& input) {
    int size = input.size();
    std::vector<cuFloatComplex> output(size);

    cuFloatComplex* d_input;
    cudaMalloc(&d_input, size*sizeof(cuFloatComplex));

    cudaMemcpy(d_input, input.data(), size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    for (int i = 0; i < size; i++) {
        std::cout << "(" << cuCrealf(input[i]) << ", " << cuCimagf(input[i]) << "i) ";
    }
    std::cout << std::endl;

    std::vector<std::vector<cuFloatComplex>> pairs = fftBreakdownHost(input);
    std::cout << "FFT Breakdown:" << std::endl;
    for (int i = 0; i < pairs.size(); i++) {
        std::cout << "Pair " << i << ": (" << cuCrealf(pairs[i][0]) << ", " << cuCimagf(pairs[i][0]) << "i) and ("
                  << cuCrealf(pairs[i][1]) << ", " << cuCimagf(pairs[i][1]) << "i)" << std::endl;
    }
    
    return input;
}

#ifdef COMPILE_MAIN
int main(int argc, char *argv[]) {
    if (argc <= 1 || argc % 2 != 1) {
        std::cout << "Usage: " << argv[0] << " <real1> <imaginary1> <real2> <imaginary2> ... <realN> <imaginaryN>\n";
        return 1;
    }

    int size = (argc - 1) / 2;
    size_t bytes = size * sizeof(float);

    float *real_values = (float*)malloc(bytes);
    float *imaginary_values = (float*)malloc(bytes);
    std::vector<cuFloatComplex> complex_values(size);

    for (int i = 0; i < 2 * size; i += 2) {
        int index = i / 2;
        complex_values[index] = make_cuFloatComplex(std::atof(argv[1 + i]), std::atof(argv[2 + i]));
    }
    
    std::vector<cuFloatComplex> fft_result = computeFFT(complex_values);

    return 0;
}
#endif
