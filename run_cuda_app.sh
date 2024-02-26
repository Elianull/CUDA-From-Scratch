#!/bin/bash

# Check if at least a file name is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <cuda-file.cu> [app arguments]"
    exit 1
fi

FILENAME=$1
BASENAME=$(basename "$FILENAME" .cu)
shift

BINDIR=./bin
mkdir -p "$BINDIR"

nvcc -o "${BINDIR}/${BASENAME}" "$FILENAME"
"${BINDIR}/${BASENAME}" "$@"