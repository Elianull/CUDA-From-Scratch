#include <iostream>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>

const float PI = 2*acos(0.0);

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

// __global__ void dftKernel(cuFloatComplex* input, cuFloatComplex* output, int size) {
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int idx = bid * blockDim.x + tid;
//     int numPairs = size / 2;

//     if (idx < numPairs) {
//         cuFloatComplex x = input[idx*2];
//         cuFloatComplex y = input[idx*2 + 1];

//         float angle = -2.0f * M_PI * idx / numPairs;
//         cuFloatComplex twiddle = make_cuFloatComplex(cos(angle), sin(angle));

//         output[idx*2] = cuCaddf(x, cuCmulf(y, twiddle));
//         output[idx*2 + 1] = cuCsubf(x, cuCmulf(y, twiddle));
//     }
// }

// __global__ void dftKernel(cuFloatComplex* d_input, cuFloatComplex* d_output, int N) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;

//     if (idx < N/2) {
//         cuFloatComplex x = d_input[idx * 2];
//         cuFloatComplex y = d_input[idx * 2 + 1];

//         cuFloatComplex sum_x = make_cuFloatComplex(0.0f, 0.0f);
//         cuFloatComplex sum_y = make_cuFloatComplex(0.0f, 0.0f);

//         for (int n = 0; n < 2; n++) {
//             float angle_x = -2.0f * M_PI * 0 * n / 2.0f;
//             float angle_y = -2.0f * M_PI * 1 * n / 2.0f;

//             cuFloatComplex exp_term_x = make_cuFloatComplex(cosf(angle_x), sinf(angle_x));
//             cuFloatComplex exp_term_y = make_cuFloatComplex(cosf(angle_y), sinf(angle_y));

//             cuFloatComplex product_x = cuCmulf(x, exp_term_x);
//             cuFloatComplex product_y = cuCmulf(y, exp_term_y);

//             sum_x = cuCaddf(sum_x, product_x);
//             sum_y = cuCaddf(sum_y, product_y);
//         }

//         d_output[idx * 2] = sum_x;
//         d_output[idx * 2 + 1] = sum_y;
//     }
// }

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

std::vector<cuFloatComplex> computeFFT(std::vector<cuFloatComplex>& input) {
    int size = input.size();
    std::vector<cuFloatComplex> output(size);

    cuFloatComplex* d_input;
    cudaMalloc(&d_input, size*sizeof(cuFloatComplex));

    cudaMemcpy(d_input, input.data(), size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    std::cout << "Input: ";
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
    
        int N = 2;  // Assuming each pair has 2 elements
    int numPairs = pairs.size();

    cuFloatComplex* dftOutput = new cuFloatComplex[N * numPairs];

    // Convert the pairs vector to a flattened array
    cuFloatComplex* flatPairs = new cuFloatComplex[N * numPairs];
    for (int i = 0; i < numPairs; i++) {
        flatPairs[i * N + 0] = pairs[i][0];
        flatPairs[i * N + 1] = pairs[i][1];
    }

    performDFT(flatPairs, dftOutput, N, numPairs);

    std::cout << "DFT Outputs: ";
    for (int i = 0; i < N * numPairs; i++) {
        std::cout << "(" << cuCrealf(dftOutput[i]) << ", " << cuCimagf(dftOutput[i]) << "i) ";
    }
    std::cout << std::endl;

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
