#include <iostream>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

__global__ void fftBreakdown(cuFloatComplex* input, cuFloatComplex* output, int size) {
    extern __shared__ cuFloatComplex sharedInput[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int numPairs = (size + 1) / 2;
    int idx = bid * blockDim.x + tid;

    if (idx < numPairs) {
        sharedInput[tid] = input[idx];
        __syncthreads();

        output[idx] = sharedInput[tid];
        if (idx + numPairs < size) {
            output[idx + numPairs] = input[idx + numPairs];
        }
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

    cuFloatComplex* d_input;
    cuFloatComplex* d_output;
    cudaMalloc(&d_input, size * sizeof(cuFloatComplex));
    cudaMalloc(&d_output, size * sizeof(cuFloatComplex));

    cudaMemcpy(d_input, input.data(), size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numPairs + blockSize - 1) / blockSize;
    fftBreakdown<<<numBlocks, blockSize, blockSize * sizeof(cuFloatComplex)>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    for (int i = 0; i < numPairs; i++) {
        cudaMemcpy(&output[i][0], &d_output[i], sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(&output[i][1], &d_output[i + numPairs], sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

__global__ void dft_kernel(cuFloatComplex* input, cuFloatComplex* output, uint32_t N, uint32_t numPairs)
{
    // Find which element of Y this thread is computing
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int p = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (k < N && p < numPairs) {
        cuFloatComplex sum = make_cuFloatComplex(0, 0);
        
        // Save the value of -2 pi * k / N
        float c = -2 * M_PI * k / N;
        
        // Each thread computes a summation containing N terms
        for (size_t n = 0; n < N; n++) {
            // e^ix = cos x + i sin x
            // Compute x[n] * exp(-2i pi * k * n / N)
            float ti, tr;
            sincosf(c * n, &ti, &tr);
            sum = cuCaddf(sum, cuCmulf(input[p * N + n], make_cuFloatComplex(tr, ti)));
        }
        
        output[p * N + k] = sum;
    }
}

// This function computes the DFT on the GPU.
void performDFT(cuFloatComplex* input, cuFloatComplex* output, uint32_t N, uint32_t numPairs)
{
    cuFloatComplex* d_input;
    cuFloatComplex* d_output;
    
    cudaMalloc((void**)&d_output, sizeof(cuFloatComplex) * N * numPairs);
    cudaMalloc((void**)&d_input, sizeof(cuFloatComplex) * N * numPairs);
    
    cudaMemcpy(d_input, input, sizeof(cuFloatComplex) * N * numPairs, cudaMemcpyHostToDevice);
    
    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device_ix);
    
    // One thread for each element of the output vector
    int block_size_x = min(N, prop.maxThreadsDim[0]);
    int block_size_y = min(numPairs, prop.maxThreadsDim[1]);
    int grid_size_x = (N + block_size_x - 1) / block_size_x;
    int grid_size_y = (numPairs + block_size_y - 1) / block_size_y;
    
    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_size_x, grid_size_y);
    
    dft_kernel<<<grid, block>>>(d_input, d_output, N, numPairs);
    
    cudaMemcpy(output, d_output, sizeof(cuFloatComplex) * N * numPairs, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void multiplyTwiddleFactors(cuFloatComplex* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n / 2; i += stride) {
        float angle = -2 * M_PI * i / n;
        cuFloatComplex twiddle = make_cuFloatComplex(cos(angle), sin(angle));

        cuFloatComplex temp = cuCmulf(data[i + n / 2], twiddle);
        data[i + n / 2] = cuCsubf(data[i], temp);
        data[i] = cuCaddf(data[i], temp);
    }
}

__global__ void bitReversalPermutation(cuFloatComplex* data, int n, int log2n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        int reversedIndex = __brev(i) >> (32 - log2n);
        if (i < reversedIndex) {
            cuFloatComplex temp = data[i];
            data[i] = data[reversedIndex];
            data[reversedIndex] = temp;
        }
    }
}

void multiplyTwiddleFactors(std::vector<cuFloatComplex>& data) {
    int n = data.size();

    for (int i = 0; i < n / 2; i++) {
        int twiddleIndex = i;
        float angle = -2 * M_PI * twiddleIndex / n;
        cuFloatComplex twiddle = make_cuFloatComplex(cos(angle), sin(angle));

        std::cout << "Twiddle factor for index " << i << ": (" << cuCrealf(twiddle) << ", " << cuCimagf(twiddle) << "i)" << std::endl;

        cuFloatComplex temp = cuCmulf(data[i + n / 2], twiddle);
        std::cout << "Multiplied element at index " << i + n / 2 << ": (" << cuCrealf(temp) << ", " << cuCimagf(temp) << "i)" << std::endl;

        data[i + n / 2] = cuCaddf(data[i + n / 2], temp);
        std::cout << "Updated element at index " << i + n / 2 << ": (" << cuCrealf(data[i + n / 2]) << ", " << cuCimagf(data[i + n / 2]) << "i)" << std::endl;

        data[i] = cuCsubf(data[i], temp);
        std::cout << "Updated element at index " << i << ": (" << cuCrealf(data[i]) << ", " << cuCimagf(data[i]) << "i)" << std::endl;

        std::cout << std::endl;
    }
}

std::vector<cuFloatComplex> computeFFT(std::vector<cuFloatComplex>& input) {
    int size = input.size();
    std::vector<cuFloatComplex> output(size);

    cuFloatComplex* d_input;
    cudaMalloc(&d_input, size * sizeof(cuFloatComplex));
    cudaMemcpy(d_input, input.data(), size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    std::cout << "Input: ";
    for (int i = 0; i < size; i++) {
        std::cout << "(" << cuCrealf(input[i]) << ", " << cuCimagf(input[i]) << "i) ";
    }
    std::cout << std::endl;

    int log2n = (int)log2(size); //log2 currently must be run on CPU

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    bitReversalPermutation<<<numBlocks, blockSize>>>(d_input, size, log2n);

    std::vector<cuFloatComplex> bitReversedData(size);
    cudaMemcpy(bitReversedData.data(), d_input, size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    std::cout << "Bit reversal: ";
    for (int i = 0; i < size; i++) {
        std::cout << "(" << cuCrealf(bitReversedData[i]) << ", " << cuCimagf(bitReversedData[i]) << "i) ";
    }
    std::cout << std::endl;

    std::vector<std::vector<cuFloatComplex>> pairs = fftBreakdownHost(bitReversedData);
    std::cout << "FFT Breakdown:" << std::endl;
    for (int i = 0; i < pairs.size(); i++) {
        std::cout << "Pair " << i << ": (" << cuCrealf(pairs[i][0]) << ", " << cuCimagf(pairs[i][0]) << "i) and ("
                  << cuCrealf(pairs[i][1]) << ", " << cuCimagf(pairs[i][1]) << "i)" << std::endl;
    }

    int N = 2;  // Assuming each pair has 2 elements
    int numPairs = pairs.size();

    cuFloatComplex* dftOutput;
    cudaMalloc(&dftOutput, N * numPairs * sizeof(cuFloatComplex));

    // Convert the pairs vector to a flattened array
    cuFloatComplex* flatPairs;
    cudaMalloc(&flatPairs, N * numPairs * sizeof(cuFloatComplex));
    for (int i = 0; i < numPairs; i++) {
        cudaMemcpy(&flatPairs[i * N], pairs[i].data(), N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    }

    performDFT(flatPairs, dftOutput, N, numPairs);

    std::cout << "DFT Outputs: ";
    for (int i = 0; i < N * numPairs; i++) {
        cuFloatComplex temp;
        cudaMemcpy(&temp, &dftOutput[i], sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        std::cout << "(" << cuCrealf(temp) << ", " << cuCimagf(temp) << "i) ";
    }
    std::cout << std::endl;

    multiplyTwiddleFactors<<<numBlocks, blockSize>>>(dftOutput, size);

    // Copy the final FFT output back to the host
    cudaMemcpy(output.data(), dftOutput, size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    //multiplyTwiddleFactors(output);

    // Free device memory
    cudaFree(d_input);
    cudaFree(dftOutput);
    cudaFree(flatPairs);

    return output;
}

#ifdef COMPILE_MAIN
int main(int argc, char *argv[]) {
    if (argc <= 1 || argc % 2 != 1) {
        std::cout << "Usage: " << argv[0] << " <real1> <imaginary1> <real2> <imaginary2> ... <realN> <imaginaryN>\n";
        return 1;
    }

    int size = (argc - 1) / 2;
    std::vector<cuFloatComplex> complex_values(size);

    for (int i = 0; i < 2 * size; i += 2) {
        int index = i / 2;
        complex_values[index] = make_cuFloatComplex(std::atof(argv[1 + i]), std::atof(argv[2 + i]));
    }

    std::vector<cuFloatComplex> fft_result = computeFFT(complex_values);

    std::cout << "FFT Outputs: ";
    for (int i = 0; i < size; i++) {
        std::cout << "(" << cuCrealf(fft_result[i]) << ", " << cuCimagf(fft_result[i]) << "i) ";
    }
    std::cout << std::endl;

    return 0;
}
#endif
