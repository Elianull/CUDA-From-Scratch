#!/bin/bash

# Check if a file name is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <cuda-file.cu>"
    exit 1
fi

FILENAME=$1

# Compile the CUDA file
nvcc -o cuda_app $FILENAME

# Run the compiled program
./cuda_app
