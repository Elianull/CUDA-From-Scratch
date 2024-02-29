#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// See https://pythonnumericalmethods.berkeley.edu/notebooks/chapter19.04-Newton-Raphson-Method.html
__device__ float sqrt_newton_raphson(float x, float tolerance, int max_iters = 1000) { // Device can only be called from device (GPU), global can be called from both
    if (x < 0) return -1.0f; // Returns error on negative num

    float guess = x / 2.0f; // Newton Raphson needs an initial guess, start with half of x
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

#ifdef COMPILE_MAIN
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num>" << std::endl;
        return 1;
    }

    int n = atoi(argv[1]);

    if (n <= 0) {
        std::cerr << "<num> must be greater than 0" << std::endl;
        return 1;
    }

    float *h_in = new float[n]; // Host input array
    float *h_out = new float[n]; // Host output array

    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(i + 1); 
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    calculate_sqrt<<<(n + 255) / 256, 256>>>(d_out, d_in, 1e-6f, n);

    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

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
#endif
