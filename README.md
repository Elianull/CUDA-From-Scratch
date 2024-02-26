## Running the Docker Container on Windows with Unix-like Paths

When executing Docker commands on Windows within a Unix-like environment (e.g., MSYS, MinGW, Cygwin), path conversion can cause issues due to the differing path formats between Windows and Unix/Linux systems. To prevent path conversion and ensure the command operates correctly, set the `MSYS_NO_PATHCONV` environment variable to `1` before running your Docker command. This approach is particularly useful when mounting volumes.

Here's how to run your Docker container with GPU support and volume mounting, treating paths in a Unix-like manner:

```bash
MSYS_NO_PATHCONV=1 docker run --rm --gpus all -v ${PWD}:/usr/src/app cuda-compiler ./run_cuda_app.sh <CUDA File> <args>
```
