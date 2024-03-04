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

echo "Cleaning old compilation files..."
rm -f "${BINDIR}"/*.o 

echo "Compiling the main CUDA file with 'COMPILE_MAIN' defined: $FILENAME"
# Compile the specified .cu file with COMPILE_MAIN defined
nvcc -D COMPILE_MAIN -c "$FILENAME" -o "${BINDIR}/${BASENAME}.o"

echo "Compiling dependency CUDA files without 'COMPILE_MAIN':"
declare -A JOBS # Associative array to hold job IDs and their corresponding commands
for file in $(find "$DIRNAME" -maxdepth 1 -type f -name '*.cu' ! -name "$(basename "$FILENAME")"); do
    OBJ_NAME="${BINDIR}/$(basename "$file" .cu).o"
    echo "Compiling $file -> $OBJ_NAME"
    nvcc -c "$file" -o "$OBJ_NAME" &
    JOBS[$!]=$file
done

# Function to check jobs and retry if necessary
check_jobs() {
    local -n jobs=$1
    local retries=3
    for job in "${!jobs[@]}"; do
        wait $job || {
            echo "Compilation failed for ${jobs[$job]}, retrying..."
            local file=${jobs[$job]}
            local OBJ_NAME="${BINDIR}/$(basename "$file" .cu).o"
            local retry=0
            while [ $retry -lt $retries ]; do
                nvcc -c "$file" -o "$OBJ_NAME"
                if [ $? -eq 0 ]; then
                    echo "Compilation succeeded for $file after retry."
                    break
                else
                    echo "Retrying compilation for $file ($((retry+1))/$retries)..."
                    retry=$((retry+1))
                fi
            done
            if [ $retry -eq $retries ]; then
                echo "Compilation failed for $file after $retries attempts."
                exit 1
            fi
        }
    done
}

# Check and retry jobs as necessary
check_jobs JOBS

echo "Linking all object files to create the executable..."
OBJFILES=$(find "$BINDIR" -type f -name '*.o')
nvcc $OBJFILES -o "${BINDIR}/${BASENAME}"

echo "Compilation complete. Executable is: ${BINDIR}/${BASENAME}"

# Execute the compiled binary with any additional arguments passed to this script
echo "Running the executable with arguments: $@"
"${BINDIR}/${BASENAME}" "$@"
