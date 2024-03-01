#!/bin/bash

echo "Starting the CUDA compilation script..."

# Check if at least a file name is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <cuda-file.cu> [app arguments]"
    exit 1
fi

FILENAME=$1
DIRNAME=$(dirname "$FILENAME")
BASENAME=$(basename "$FILENAME" .cu)
shift

# Directory to place the compiled binary
BINDIR=./bin
mkdir -p "$BINDIR"

echo "Compiling the main CUDA file with 'COMPILE_MAIN' defined: $FILENAME"
# Compile the specified .cu file with COMPILE_MAIN defined
nvcc -D COMPILE_MAIN -c "$FILENAME" -o "${BINDIR}/${BASENAME}.o"

# Find and compile other .cu files as dependencies, excluding the main .cu file, without defining COMPILE_MAIN
echo "Compiling dependency CUDA files without 'COMPILE_MAIN':"
for file in $(find "$DIRNAME" -maxdepth 1 -type f -name '*.cu' ! -name "$(basename "$FILENAME")"); do
    OBJ_NAME="${BINDIR}/$(basename "$file" .cu).o"
    echo "Compiling $file -> $OBJ_NAME"
    nvcc -c "$file" -o "$OBJ_NAME" &
done

# Link all object files including the main .cu file object
echo "Linking all object files to create the executable..."
OBJFILES=$(find "$BINDIR" -type f -name '*.o')
nvcc $OBJFILES -o "${BINDIR}/${BASENAME}"

echo "Compilation complete. Executable is: ${BINDIR}/${BASENAME}"

# Execute the compiled binary with any additional arguments passed to this script
echo "Running the executable with arguments: $@"
"${BINDIR}/${BASENAME}" "$@"
