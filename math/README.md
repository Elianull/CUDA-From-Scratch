# CUDA Programming: Function Implementations and Lessons Learned



This document outlines the CUDA functions I've implemented, focusing on key takeaways and lessons learned from each.



### Summation


- **Functionality**: Implements discrete summation to calculate averages.

- **Lessons Learned**:

  - Memory allocation across devices was an initially painful experience.



### Product


- **Functionality**: Calculates the product of an array.

- **Lessons Learned**:

  - Was stuck debugging for a while, the identity element ended up being the issue.



### Mean


- **Functionality**: Implements a discrete working average.

- **Lessons Learned**:

  - Using previously created methods proved to be more difficult than expected. For now I've just reused the code as I'd like to keep things within my script, and automatically linking them won't work thanks to main functions even wrapped within compiler variables.



### Square root


- **Functionality**: Calculates the square root using the Newton-Raphson method.

- **Lessons Learned**:

  - Newton-Raphson!

  - First run with making functions massively parallel

  - Reached a point of needing to make performance precision tradeoffs.

 

### Linear regression


- **Functionality**: Performs linear regression via the Least Squares method.

- **Lessons Learned**:

  - Least squares!

  - Dependencies here put me through the ringer. Ended up modifying my compilation script to automatically resolve that, using compiler variables to allow me to keep the main methods for testing.

  - Synchronizing memory across many kernels



### Dot product


- **Functionality**: Calculates dot product.

- **Lessons Learned**:

  - The first kernel implementation that did not require me to learn anything to implement



### Matrix multiplication


- **Functionality**: Performs matrix multiplication using shared memory and tile-based decomposition.

- **Lessons Learned**:

  - First implementation of a function that had a delay in response time, thus leading me to optimization.
 
  - Shared memory implementation marked a significant speed improvement
 
  - Debugging shared memory proved difficult when everything is not perfectly in sync



### Standard deviation


- **Functionality**: Calculates standard deviation

- **Lessons Learned**:

  - Overall straightforward, but an important review in parallel reduction
    
  - Compilation script fixes in the case of dependencies



### Fast Fourier Transform (FFT)


- **Functionality**:  Implements the Cooley-Tukey FFT algorithm for complex-valued input.

- **Lessons Learned**:
  - The Cooley-Tukey algorithm and butterfly operations took some time to fully grasp and implement correctly.
  
  - Bit-reversal permutation was a tricky step, requiring careful indexing and bit manipulation.
  
  - Precalculating twiddle factors and storing them in constant memory provided a noticeable performance boost.
  
  - Dividing the FFT into smaller sub-problems and utilizing shared memory greatly improved performance.
  
  - Debugging complex-valued operations and ensuring correct indexing throughout the FFT stages was a challenge.
  
  - Finding the optimal balance between parallelism and memory usage required experimentation with different thread block sizes and shared memory configurations.
  
  - Handling non-power-of-two input sizes required additional padding or truncation logic.
  
  - Comparing CUDA FFT results with a CPU reference implementation was crucial for validating correctness.

## TODOs


- Parallel division
- Vector addition
- Scalar multiplication
- Vector normalization
- Eigenvalue calculation
- Eigenvector calculation
- Random number generation
- Black scholes equation
