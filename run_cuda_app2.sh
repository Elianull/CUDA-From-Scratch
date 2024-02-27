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

# Initialize an empty string to keep track of the dependency files
DEP_FILES=""

# Function to find and add .cu file dependencies
function find_dependencies() {
    local file=$1
    while IFS= read -r line; do
        if [[ $line =~ \#include\ \"(.*)\" ]]; then
            local inc_file=$(dirname "$file")/"${BASH_REMATCH[1]}"
            # Check if the included file is a .cu file and add it if so
            if [[ $inc_file == *.cu ]]; then
                # Add the file to the list if it's not already included
                if [[ ! " $DEP_FILES " =~ " $inc_file " ]]; then
                    DEP_FILES+=" $inc_file"
                    # Recursively find dependencies in the newly added file
                    find_dependencies "$inc_file"
                fi
            fi
        fi
    done < "$file"
}

# Find dependencies in the main .cu file
find_dependencies "$FILENAME"

# Compile each .cu file to an object file
OBJ_FILES=""
for file in $FILENAME $DEP_FILES; do
    OBJ_FILE="${file%.cu}.o"
    nvcc -c "$file" -o "$OBJ_FILE"
    OBJ_FILES+=" $OBJ_FILE"
done

# Link object files together
nvcc -o "${BINDIR}/${BASENAME}" $OBJ_FILES

# Clean up object files
rm $OBJ_FILES

# Execute the compiled binary with any additional arguments
"${BINDIR}/${BASENAME}" "$@"
