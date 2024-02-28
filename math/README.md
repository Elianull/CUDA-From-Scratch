# CUDA Programming: Function Implementations and Lessons Learned



This document outlines the CUDA functions I've implemented, focusing on key takeaways and lessons learned from each.



## summation.cu



- **Functionality**: Implements discrete summation to calculate averages.

- **Lessons Learned**:

  - Memory allocation across devices was an initially painful experience.



## product.cu



- **Functionality**: Calculates the product of an array.

- **Lessons Learned**:

  - Was stuck debugging for a while, the identity element ended up being the issue.



## mean.cu



- **Functionality**: Implements a discrete working average.

- **Lessons Learned**:

  - Using previously created methods proved to be more difficult than expected. For now I've just reused the code as I'd like to keep things within my script, and automatically linking them won't work thanks to main functions even wrapped within compiler variables.



## sqrt.cu



- **Functionality**: Calculates the square root using the Newton-Raphson method.

- **Lessons Learned**:

  - Newton-Raphson!

  - First run with making functions massively parallel

  - Reached a point of needing to make performance precision tradeoffs.

