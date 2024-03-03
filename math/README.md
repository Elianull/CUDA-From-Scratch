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


- **Functionality**: Calculates dot product

- **Lessons Learned**:

  - The first kernel implementation that did not require me to learn anything to implement



## TODOs


- Parallel division
- Vector addition
- Scalar multiplication
- Matrix multiplication
- Vector normalization
- Fast fourier transform
- Eigenvalue calculation
- Eigenvector calculation
- Random number generation
- Black scholes equation
