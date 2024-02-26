FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04

# Install build essentials and CUDA toolkit
RUN apt-get update && apt-get install -y \
    build-essential \
    cuda-toolkit-11-2

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the script that will compile and run the CUDA program
COPY run_cuda_app.sh /usr/src/app

# Make the script executable
RUN chmod +x /usr/src/app/run_cuda_app.sh

# Command to run on container start (this script expects a file name as an argument)
CMD ["./run_cuda_app.sh"]