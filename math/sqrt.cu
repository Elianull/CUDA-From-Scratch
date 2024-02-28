#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// See https://pythonnumericalmethods.berkeley.edu/notebooks/chapter19.04-Newton-Raphson-Method.html
__device__ float sqrt_newton_raphson(float x, float tolerance, int max_iters = 1000) { // Device can only be called from device (GPU), global can be called from both
    if (x < 0) return -1.0f; // Error for negative numbers

    float guess = x / 2.0f;
    int iter = 0;
    while (abs(guess * guess - x) > tolerance && iter < max_iters) {
        guess = (guess + x / guess) / 2.0f;
        iter++;
    }
    return guess;
}

__global__ void calculate_sqrt(float *d_out, float *d_in, float tolerance, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        d_out[idx] = sqrt_newton_raphson(d_in[idx], tolerance);
    }
}


int main(int argc, char *argv[]) {
    // Check if a command-line argument is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num>" << std::endl;
        return 1;
    }

    // Convert the first command-line argument to an integer
    int n = atoi(argv[1]); // Number of elements

    if (n <= 0) {
        std::cerr << "Please enter a positive integer for <num>." << std::endl;
        return 1;
    }

    float *h_in = new float[n]; // Host input array
    float *h_out = new float[n]; // Host output array

    // Initialize input data
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(i + 1); // Start from 1 to avoid sqrt(0)
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate square root for each element
    calculate_sqrt<<<(n + 255) / 256, 256>>>(d_out, d_in, 1e-6f, n);

    // Copy the result back to the host
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the results (optional)
    for (int i = 0; i < n; ++i) {
        std::cout << "sqrt(" << h_in[i] << ") = " << h_out[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Free host memory
    delete[] h_in;
    delete[] h_out;

    return 0;
}
