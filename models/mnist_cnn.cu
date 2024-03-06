#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "../loss/cross_entropy_loss.cuh"
#include "../math/matmul.cuh"
#include "../math/relu.cuh"
#include "../math/softmax.cuh"
#include "../math/sigmoid.cuh"

#ifdef COMPILE_MAIN

#define cudaCheckError(...) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d", cudaGetErrorString(err), __FILE__, __LINE__); \
        if (strcmp("", #__VA_ARGS__) != 0) { \
            fprintf(stderr, " - %s", #__VA_ARGS__); \
        } \
        fprintf(stderr, "\n"); \
    } \
}

struct MNISTDataset {
    std::vector<std::vector<float>> images; // Normalized pixel values
    std::vector<int> labels;   // Labels
};

MNISTDataset loadData(const std::string& filename) {
    MNISTDataset dataset;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1); // Or handle the error as appropriate
    }
    std::string line;

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> image(28*28); // MNIST images are 28x28
        int label;
        int pixelIdx = 0;

        // Read label
        std::getline(lineStream, cell, ',');
        label = std::stoi(cell);
        dataset.labels.push_back(label);

        // Read pixels
        float mean = 0.1307f;
        float stddev = 0.3081f;
        while (std::getline(lineStream, cell, ',')) {
            float pixel = (std::stof(cell) / 255.0f - mean) / stddev;
            image[pixelIdx++] = pixel;
        }

        dataset.images.push_back(image);
        //dataset.images.insert(dataset.images.end(), image.begin(), image.end());
    }

    return dataset;
}

inline void printDigitImage(const std::vector<float>& image) {
    const int width = 28; // MNIST images are 28x28
    const int height = 28;
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float pixelValue = image[i * width + j];
            // Map the pixel value to a character
            char pixelChar = ' ';
            if (pixelValue > 0.75f) {
                pixelChar = '#';
            } else if (pixelValue > 0.5f) {
                pixelChar = '+';
            } else if (pixelValue > 0.25f) {
                pixelChar = '.';
            }
            std::cout << pixelChar;
        }
        std::cout << std::endl;
    }
}

__global__ void denseLayerKernel(float* input, float* weights, float* biases, float* output, int numRows, int numCols, int numOutput) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numOutput) {
        float sum = 0.0;
        for (int i = 0; i < numCols; ++i) {
            float inputVal = input[row * numCols + i];
            float weightVal = weights[i * numOutput + col];
            sum += inputVal * weightVal;
            //printf("denseLayerKernel: input[%d] = %f, weights[%d] = %f\n", row * numCols + i, inputVal, i * numOutput + col, weightVal);
        }
        
        // Apply ReLU activation function
        output[row * numOutput + col] = max(0.0f, sum + biases[col]);
        
        //printf("denseLayerKernel: biases[%d] = %f, output[%d] = %f\n", col, biases[col], row * numOutput + col, output[row * numOutput + col]);
    }
}

__global__ void convLayerKernel(float* input, float* weights, float* biases, float* output, int numImages, int inputHeight, int inputWidth, int numChannels, int numFilters, int filterSize, int stride, int padding) {
    int outputHeight = (inputHeight - filterSize + 2 * padding) / stride + 1;
    int outputWidth = (inputWidth - filterSize + 2 * padding) / stride + 1;

    int image = blockIdx.z * blockDim.z + threadIdx.z;
    int filter = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (image < numImages && filter < numFilters && row < outputHeight) {
        for (int col = 0; col < outputWidth; ++col) {
            float sum = 0.0;
            for (int c = 0; c < numChannels; ++c) {
                for (int i = 0; i < filterSize; ++i) {
                    for (int j = 0; j < filterSize; ++j) {
                        int inputRow = row * stride - padding + i;
                        int inputCol = col * stride - padding + j;
                        if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                            sum += input[image * inputHeight * inputWidth * numChannels + c * inputHeight * inputWidth + inputRow * inputWidth + inputCol] *
                                   weights[filter * numChannels * filterSize * filterSize + c * filterSize * filterSize + i * filterSize + j];
                        }
                    }
                }
            }
            output[image * numFilters * outputHeight * outputWidth + filter * outputHeight * outputWidth + row * outputWidth + col] = sum + biases[filter];
        }
    }
}

__global__ void convLayerBackwardKernel(float* d_images, float* d_convWeights, float* d_gradConvOutput, float* d_gradConvWeights, float* d_gradConvBiases,
                                        int batchSize, int inputHeight, int inputWidth, int numChannels,
                                        int numFilters, int filterSize, int stride, int padding) {
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;
    int filterIdx = blockIdx.z;

    int convOutputHeight = (inputHeight - filterSize + 2 * padding) / stride + 1;
    int convOutputWidth = (inputWidth - filterSize + 2 * padding) / stride + 1;

    if (outputRow < convOutputHeight && outputCol < convOutputWidth && filterIdx < numFilters) {
        float gradBias = 0.0f;

        for (int batch = 0; batch < batchSize; ++batch) {
            float gradOutput = d_gradConvOutput[batch * numFilters * convOutputHeight * convOutputWidth +
                                                filterIdx * convOutputHeight * convOutputWidth +
                                                outputRow * convOutputWidth + outputCol];

            for (int channel = 0; channel < numChannels; ++channel) {
                for (int i = 0; i < filterSize; ++i) {
                    for (int j = 0; j < filterSize; ++j) {
                        int inputRow = outputRow * stride - padding + i;
                        int inputCol = outputCol * stride - padding + j;

                        if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                            float inputVal = d_images[batch * numChannels * inputHeight * inputWidth +
                                                      channel * inputHeight * inputWidth +
                                                      inputRow * inputWidth + inputCol];

                            atomicAdd(&d_gradConvWeights[filterIdx * numChannels * filterSize * filterSize +
                                                         channel * filterSize * filterSize +
                                                         i * filterSize + j], gradOutput * inputVal);
                        }
                    }
                }
            }

            gradBias += gradOutput;
        }

        atomicAdd(&d_gradConvBiases[filterIdx], gradBias);
    }
}

__global__ void convertLabelsToOneHot(int* labels, float* targets, int batchSize, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        int label = labels[idx];
        //printf("convertLabelsToOneHot: labels[%d] = %d\n", idx, label);
        for (int c = 0; c < numClasses; ++c) {
            targets[idx * numClasses + c] = (c == label) ? 1.0f : 0.0f;
        }
        //printf("convertLabelsToOneHot: targets[%d] = %f\n", idx * numClasses + label, targets[idx * numClasses + label]);
    }
}

__global__ void softmaxGradKernel(float* output, float* targets, float* gradOutput, int batchSize, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * numClasses) {
        int sample = idx / numClasses;
        int c = idx % numClasses;
        float outputVal = output[idx];
        float targetVal = targets[sample * numClasses + c];
        gradOutput[idx] = (outputVal - targetVal) / batchSize;
        //printf("softmaxGradKernel: output[%d] = %f, targets[%d] = %f, gradOutput[%d] = %f\n", idx, outputVal, sample * numClasses + c, targetVal, idx, gradOutput[idx]);
    }
}

__global__ void reluGradKernel(float* output, float* gradOutput, float* gradInput, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradInput[idx] = (output[idx] > 0) ? gradOutput[idx] : 0;
        //printf("reluGradKernel: gradInput[%d] = %f\n", idx, gradInput[idx]); // Debugging statement
    }
}

__global__ void biasGradientKernel(float* gradOutput, float* gradBiases, int batchSize, int numOutput) {
    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx < numOutput) {
        float gradBias = 0.0f;
        for (int sample = 0; sample < batchSize; ++sample) {
            gradBias += gradOutput[sample * numOutput + outputIdx];
        }
        gradBiases[outputIdx] = gradBias;
        //printf("biasGradientKernel: gradBiases[%d] = %f\n", outputIdx, gradBiases[outputIdx]); // Debugging statement
    }
}

__global__ void updateWeightsAndBiasesKernel(float* weights, float* biases, float* gradWeights, float* gradBiases, int inputSize, int numOutput, float learningRate) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < inputSize && col < numOutput) {
        int idx = row * numOutput + col;

        float clipValue = 1.0f;
        gradWeights[idx] = max(min(gradWeights[idx], clipValue), -clipValue);
        gradBiases[col] = max(min(gradBiases[col], clipValue), -clipValue);

        weights[idx] -= learningRate * gradWeights[idx];
        if (row == 0) {
            biases[col] -= learningRate * gradBiases[col];
        }
        // printf("updateWeightsAndBiasesKernel: weights[%d] = %f, biases[%d] = %f\n", idx, weights[idx], col, biases[col]); // Debugging statement
    }
}

__global__ void denseLayerGradientKernel(float* input, float* gradOutput, float* gradWeights, int batchSize, int inputSize, int numOutput) {
    extern __shared__ float sharedMem[];
    int chunkSize = (inputSize + blockDim.x - 1) / blockDim.x;
    int numChunks = (inputSize + chunkSize - 1) / chunkSize;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float gradWeight = 0.0f;

    for (int chunk = 0; chunk < numChunks; ++chunk) {
        int startIdx = chunk * chunkSize;
        int endIdx = min(static_cast<int>((chunk + 1) * chunkSize), inputSize);
        int chunkElements = endIdx - startIdx;

        float* sharedInput = &sharedMem[0];
        float* sharedGradOutput = &sharedMem[chunkElements * blockDim.x];

        for (int batch = 0; batch < batchSize; batch += blockDim.x) {
            int inputIdx = startIdx + (batch + threadIdx.x) * inputSize;
            int gradOutputIdx = (batch + threadIdx.x) * numOutput + col;

            if (inputIdx < (batch + threadIdx.x + 1) * inputSize && col < numOutput && batch + threadIdx.x < batchSize) {
                sharedInput[threadIdx.y * blockDim.x + threadIdx.x] = input[inputIdx + row];
                sharedGradOutput[threadIdx.y * blockDim.x + threadIdx.x] = gradOutput[gradOutputIdx];
            } else {
                sharedInput[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
                sharedGradOutput[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
            }

            __syncthreads();

            for (int i = 0; i < blockDim.x; ++i) {
                gradWeight += sharedInput[threadIdx.y * blockDim.x + i] * sharedGradOutput[i * blockDim.y + threadIdx.x];
            }

            __syncthreads();
        }
    }

    if (row < inputSize && col < numOutput) {
        gradWeights[row * numOutput + col] = gradWeight;
    }
}

__global__ void reluActivationKernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = max(0.0f, input[idx]);
    }
}

__global__ void maxPoolingKernel(float* input, float* output, int numImages, int inputHeight, int inputWidth, int numChannels, int poolSize, int stride) {
    int outputHeight = (inputHeight - poolSize) / stride + 1;
    int outputWidth = (inputWidth - poolSize) / stride + 1;

    int image = blockIdx.z * blockDim.z + threadIdx.z;
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (image < numImages && channel < numChannels && row < outputHeight) {
        for (int col = 0; col < outputWidth; ++col) {
            float maxVal = -1e38;
            for (int i = 0; i < poolSize; ++i) {
                for (int j = 0; j < poolSize; ++j) {
                    int inputRow = row * stride + i;
                    int inputCol = col * stride + j;
                    if (inputRow < inputHeight && inputCol < inputWidth) {
                        maxVal = max(maxVal, input[image * inputHeight * inputWidth * numChannels + channel * inputHeight * inputWidth + inputRow * inputWidth + inputCol]);
                    }
                }
            }
            output[image * numChannels * outputHeight * outputWidth + channel * outputHeight * outputWidth + row * outputWidth + col] = maxVal;
        }
    }
}

__global__ void maxPoolingBackwardKernel(float* d_poolOutput, float* d_gradPoolOutput, float* d_gradConvOutput,
                                         int batchSize, int convOutputHeight, int convOutputWidth, int numFilters,
                                         int poolSize, int poolStride) {
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;
    int filterIdx = blockIdx.z;

    if (outputRow < convOutputHeight && outputCol < convOutputWidth && filterIdx < numFilters) {
        for (int batch = 0; batch < batchSize; ++batch) {
            int poolRow = outputRow / poolStride;
            int poolCol = outputCol / poolStride;

            if (poolRow < (convOutputHeight - poolSize + 1) / poolStride && poolCol < (convOutputWidth - poolSize + 1) / poolStride) {
                float maxVal = d_poolOutput[batch * numFilters * ((convOutputHeight - poolSize + 1) / poolStride) * ((convOutputWidth - poolSize + 1) / poolStride) +
                                            filterIdx * ((convOutputHeight - poolSize + 1) / poolStride) * ((convOutputWidth - poolSize + 1) / poolStride) +
                                            poolRow * ((convOutputWidth - poolSize + 1) / poolStride) + poolCol];

                float convVal = d_gradConvOutput[batch * numFilters * convOutputHeight * convOutputWidth +
                                                 filterIdx * convOutputHeight * convOutputWidth +
                                                 outputRow * convOutputWidth + outputCol];

                if (convVal == maxVal) {
                    d_gradConvOutput[batch * numFilters * convOutputHeight * convOutputWidth +
                                     filterIdx * convOutputHeight * convOutputWidth +
                                     outputRow * convOutputWidth + outputCol] = d_gradPoolOutput[batch * numFilters * ((convOutputHeight - poolSize + 1) / poolStride) * ((convOutputWidth - poolSize + 1) / poolStride) +
                                                                                                filterIdx * ((convOutputHeight - poolSize + 1) / poolStride) * ((convOutputWidth - poolSize + 1) / poolStride) +
                                                                                                poolRow * ((convOutputWidth - poolSize + 1) / poolStride) + poolCol];
                }
            }
        }
    }
}

void initializeWeightsAndBiases(float* d_weights, float* d_biases, int inputSize, int numOutput) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    
    unsigned long long seed = 973;
    printf("Seed value: %llu\n", seed);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    float stddev = sqrt(2.0f / inputSize);
    printf("Standard deviation for weight initialization: %f\n", stddev);
    curandGenerateNormal(gen, d_weights, inputSize * numOutput, 0.0f, stddev);
    
    cudaMemset(d_biases, 0, numOutput * sizeof(float));

    curandDestroyGenerator(gen);
}

void trainModel(MNISTDataset& trainData, int epochs, int batchSize, float learningRate) {
    // Define the dimensions and hyperparameters for the convolutional and pooling layers
    int inputHeight = 28;
    int inputWidth = 28;
    int numChannels = 1;
    int numFilters = 16;
    int filterSize = 5;
    int stride = 1;
    int padding = 2;
    int poolSize = 2;

    // Calculate the output dimensions of the convolutional and pooling layers
    int convOutputHeight = (inputHeight - filterSize + 2 * padding) / stride + 1;
    int convOutputWidth = (inputWidth - filterSize + 2 * padding) / stride + 1;
    int poolOutputHeight = (convOutputHeight - poolSize) / poolSize + 1;
    int poolOutputWidth = (convOutputWidth - poolSize) / poolSize + 1;

    // Update the hidden layer size based on the output of the pooling layer
    int hiddenSize = poolOutputHeight * poolOutputWidth * numFilters;
    int numOutput = 10;

    // Allocate device memory for the convolutional and pooling layers
    float *d_convWeights, *d_convBiases, *d_convOutput, *d_poolOutput;
    float *d_gradConvWeights, *d_gradConvBiases, *d_gradConvOutput, *d_gradPoolOutput;
    cudaMalloc(&d_convWeights, numFilters * numChannels * filterSize * filterSize * sizeof(float));
    cudaMalloc(&d_convBiases, numFilters * sizeof(float));
    cudaMalloc(&d_convOutput, batchSize * numFilters * convOutputHeight * convOutputWidth * sizeof(float));
    cudaMalloc(&d_poolOutput, batchSize * numFilters * poolOutputHeight * poolOutputWidth * sizeof(float));
    cudaMalloc(&d_gradConvWeights, numFilters * numChannels * filterSize * filterSize * sizeof(float));
    cudaMalloc(&d_gradConvBiases, numFilters * sizeof(float));
    cudaMalloc(&d_gradConvOutput, batchSize * numFilters * convOutputHeight * convOutputWidth * sizeof(float));
    cudaMalloc(&d_gradPoolOutput, batchSize * numFilters * poolOutputHeight * poolOutputWidth * sizeof(float));
    cudaCheckError("Conv malloc");

    // Initialize the convolutional weights and biases
    initializeWeightsAndBiases(d_convWeights, d_convBiases, numChannels * filterSize * filterSize, numFilters);
    cudaCheckError("Initialize conv weights and biases");

    // Allocate device memory for the fully connected layers
    float *d_hiddenWeights, *d_hiddenBiases, *d_hiddenOutput;
    float *d_weights, *d_biases, *d_output, *d_images, *d_targets;
    float *d_gradHiddenWeights, *d_gradHiddenBiases, *d_gradHiddenOutput;
    float *d_gradWeights, *d_gradBiases, *d_gradOutput;
    int *d_labels;

    std::vector<float> flattenedImages;
    for (const auto& image : trainData.images) {
        flattenedImages.insert(flattenedImages.end(), image.begin(), image.end());
    }
    //cudaCheckError("Flatten");

    cudaMalloc(&d_hiddenWeights, hiddenSize * numOutput * sizeof(float));
    cudaMalloc(&d_hiddenBiases, numOutput * sizeof(float));
    cudaMalloc(&d_hiddenOutput, batchSize * numOutput * sizeof(float));
    cudaMalloc(&d_weights, hiddenSize * numOutput * sizeof(float));
    cudaMalloc(&d_biases, numOutput * sizeof(float));
    cudaMalloc(&d_output, batchSize * numOutput * sizeof(float));
    cudaMalloc(&d_images, flattenedImages.size() * sizeof(float));
    cudaMalloc(&d_labels, trainData.labels.size() * sizeof(int));
    cudaMalloc(&d_targets, batchSize * numOutput * sizeof(float));
    cudaMalloc(&d_gradHiddenWeights, hiddenSize * numOutput * sizeof(float));
    cudaMalloc(&d_gradHiddenBiases, numOutput * sizeof(float));
    cudaMalloc(&d_gradHiddenOutput, batchSize * numOutput * sizeof(float));
    cudaMalloc(&d_gradWeights, hiddenSize * numOutput * sizeof(float));
    cudaMalloc(&d_gradBiases, numOutput * sizeof(float));
    cudaMalloc(&d_gradOutput, batchSize * numOutput * sizeof(float));
    //cudaCheckError("Fully connected malloc");

    // Initialize the fully connected weights and biases
    initializeWeightsAndBiases(d_hiddenWeights, d_hiddenBiases, hiddenSize, numOutput);
    initializeWeightsAndBiases(d_weights, d_biases, hiddenSize, numOutput);
    //cudaCheckError("Initialize fully connected weights and biases");

    std::cout << "d_images: " << d_images << std::endl;
    std::cout << "flattenedImages.data(): " << flattenedImages.data() << std::endl;
    std::cout << "flattenedImages.size(): " << flattenedImages.size() << std::endl;
    std::cout << "sizeof(float): " << sizeof(float) << std::endl;
    cudaMemcpy(d_images, flattenedImages.data(), flattenedImages.size() * sizeof(float), cudaMemcpyHostToDevice);
    //cudaCheckError("Copy images to device");

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0;
        int numBatches = trainData.images.size() / batchSize;

        for (int batch = 0; batch < numBatches; ++batch) {
            int offset = batch * batchSize;

            cudaMemcpy(d_images + offset * inputHeight * inputWidth, &flattenedImages[offset * inputHeight * inputWidth], batchSize * inputHeight * inputWidth * sizeof(float), cudaMemcpyHostToDevice);
            //cudaCheckError("Copy batch images to device");
            cudaMemcpy(d_labels, &trainData.labels[offset], batchSize * sizeof(int), cudaMemcpyHostToDevice);
            //cudaCheckError("Copy batch labels to device");

            // Forward pass
            dim3 convThreadsPerBlock(8, 8, 8);
            dim3 convNumBlocks((convOutputHeight + convThreadsPerBlock.x - 1) / convThreadsPerBlock.x,
                               (numFilters + convThreadsPerBlock.y - 1) / convThreadsPerBlock.y,
                               (batchSize + convThreadsPerBlock.z - 1) / convThreadsPerBlock.z);
            convLayerKernel<<<convNumBlocks, convThreadsPerBlock>>>(d_images + offset * inputHeight * inputWidth, d_convWeights, d_convBiases, d_convOutput,
                                                                    batchSize, inputHeight, inputWidth, numChannels,
                                                                    numFilters, filterSize, stride, padding);
            //cudaCheckError("Convolutional layer");
            reluActivationKernel<<<(batchSize * numFilters * convOutputHeight * convOutputWidth + 255) / 256, 256>>>(d_convOutput, batchSize * numFilters * convOutputHeight * convOutputWidth);
            //cudaCheckError("ReLU activation");

            dim3 poolThreadsPerBlock(8, 8, 8);
            dim3 poolNumBlocks((poolOutputHeight + poolThreadsPerBlock.x - 1) / poolThreadsPerBlock.x,
                               (numFilters + poolThreadsPerBlock.y - 1) / poolThreadsPerBlock.y,
                               (batchSize + poolThreadsPerBlock.z - 1) / poolThreadsPerBlock.z);
            maxPoolingKernel<<<poolNumBlocks, poolThreadsPerBlock>>>(d_convOutput, d_poolOutput,
                                                                     batchSize, convOutputHeight, convOutputWidth, numFilters,
                                                                     poolSize, poolSize);
            //cudaCheckError("Max pooling");

            dim3 hiddenThreadsPerBlock(16, 16);
            dim3 hiddenNumBlocks((batchSize + hiddenThreadsPerBlock.x - 1) / hiddenThreadsPerBlock.x,
                                 (numOutput + hiddenThreadsPerBlock.y - 1) / hiddenThreadsPerBlock.y);
            denseLayerKernel<<<hiddenNumBlocks, hiddenThreadsPerBlock>>>(d_poolOutput, d_hiddenWeights, d_hiddenBiases, d_hiddenOutput,
                                                                         batchSize, hiddenSize, numOutput);
            //cudaCheckError("Hidden dense layer");
            reluActivationKernel<<<(batchSize * numOutput + 255) / 256, 256>>>(d_hiddenOutput, batchSize * numOutput);
            //cudaCheckError("ReLU activation");

            denseLayerKernel<<<hiddenNumBlocks, hiddenThreadsPerBlock>>>(d_hiddenOutput, d_weights, d_biases, d_output,
                                                                         batchSize, numOutput, numOutput);
            //cudaCheckError("Output dense layer");
            softmax(d_output, d_output, batchSize * numOutput);
            //cudaCheckError("Softmax activation");

            convertLabelsToOneHot<<<(batchSize + 255) / 256, 256>>>(d_labels, d_targets, batchSize, numOutput);
            //cudaCheckError("Convert labels to one-hot");

            // Compute loss
            float batchLoss = crossEntropyLoss(d_output, d_targets, numOutput, batchSize, batchSize);
            //cudaCheckError("Compute loss");
            totalLoss += batchLoss;

            // Backward pass
            softmaxGradKernel<<<hiddenNumBlocks, hiddenThreadsPerBlock>>>(d_output, d_targets, d_gradOutput, batchSize, numOutput);
            //cudaCheckError("Softmax gradient");
            int outputSharedMemSize = min(49152, static_cast<int>((numOutput + numOutput) * hiddenThreadsPerBlock.x * sizeof(float)));
            denseLayerGradientKernel<<<hiddenNumBlocks, hiddenThreadsPerBlock, outputSharedMemSize>>>(d_hiddenOutput, d_gradOutput, d_gradWeights, batchSize, numOutput, numOutput);
            //cudaCheckError("Output dense layer gradient");
            biasGradientKernel<<<hiddenNumBlocks.x, numOutput>>>(d_gradOutput, d_gradBiases, batchSize, numOutput);
            //cudaCheckError("Output bias gradient");

            reluGradKernel<<<(batchSize * numOutput + 255) / 256, 256>>>(d_hiddenOutput, d_gradHiddenOutput, d_gradHiddenOutput, batchSize * numOutput);
            //cudaCheckError("ReLU gradient");
            int hiddenSharedMemSize = min(49152, static_cast<int>((hiddenSize + numOutput) * hiddenThreadsPerBlock.x * sizeof(float)));
            denseLayerGradientKernel<<<hiddenNumBlocks, hiddenThreadsPerBlock, hiddenSharedMemSize>>>(d_poolOutput, d_gradHiddenOutput, d_gradHiddenWeights, batchSize, hiddenSize, numOutput);
            //cudaCheckError("Hidden dense layer gradient");
            biasGradientKernel<<<hiddenNumBlocks.x, numOutput>>>(d_gradHiddenOutput, d_gradHiddenBiases, batchSize, numOutput);
            //cudaCheckError("Hidden bias gradient");


            // Backpropagate through max pooling
            maxPoolingBackwardKernel<<<poolNumBlocks, poolThreadsPerBlock>>>(d_convOutput, d_gradPoolOutput, d_gradConvOutput,
                                                                             batchSize, convOutputHeight, convOutputWidth, numFilters,
                                                                             poolSize, poolSize);
            //cudaCheckError("Max pooling backward");

            // Backpropagate through convolutional layer
            reluGradKernel<<<(batchSize * numFilters * convOutputHeight * convOutputWidth + 255) / 256, 256>>>(d_convOutput, d_gradConvOutput, d_gradConvOutput, batchSize * numFilters * convOutputHeight * convOutputWidth);
            //cudaCheckError("ReLU gradient");
            convLayerBackwardKernel<<<convNumBlocks, convThreadsPerBlock>>>(d_images + offset * inputHeight * inputWidth, d_convWeights, d_gradConvOutput, d_gradConvWeights, d_gradConvBiases,
                                                                            batchSize, inputHeight, inputWidth, numChannels,
                                                                            numFilters, filterSize, stride, padding);
            //cudaCheckError("Convolutional layer backward");

            // Update weights and biases
            updateWeightsAndBiasesKernel<<<hiddenNumBlocks, hiddenThreadsPerBlock>>>(d_hiddenWeights, d_hiddenBiases, d_gradHiddenWeights, d_gradHiddenBiases, hiddenSize, numOutput, learningRate);
            //cudaCheckError("Update hidden weights and biases");

            updateWeightsAndBiasesKernel<<<hiddenNumBlocks, hiddenThreadsPerBlock>>>(d_weights, d_biases, d_gradWeights, d_gradBiases, numOutput, numOutput, learningRate);
            //cudaCheckError("Update output weights and biases");

            updateWeightsAndBiasesKernel<<<convNumBlocks, convThreadsPerBlock>>>(d_convWeights, d_convBiases, d_gradConvWeights, d_gradConvBiases, numChannels * filterSize * filterSize, numFilters, learningRate);
            //cudaCheckError("Update convolutional weights and biases");
        }

        std::cout << "Epoch " << epoch << ": Average Loss = " << totalLoss / numBatches << std::endl;
    }

    // Cleanup: Free device memory
    cudaFree(d_hiddenWeights);
    cudaFree(d_hiddenBiases);
    cudaFree(d_hiddenOutput);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_targets);
    cudaFree(d_gradHiddenWeights);
    cudaFree(d_gradHiddenBiases);
    cudaFree(d_gradHiddenOutput);
    cudaFree(d_gradWeights);
    cudaFree(d_gradBiases);
    cudaFree(d_gradOutput);
    cudaFree(d_convWeights);
    cudaFree(d_convBiases);
    cudaFree(d_convOutput);
    cudaFree(d_poolOutput);
    cudaFree(d_gradConvWeights);
    cudaFree(d_gradConvBiases);
    cudaFree(d_gradConvOutput);
    cudaFree(d_gradPoolOutput);
}

int main() {
    std::string trainFile = "data/mnist_train.csv";
    std::string testFile = "data/mnist_test.csv";

    // Read and preprocess the training data
    MNISTDataset trainData = loadData(trainFile);

    if (trainData.images.empty() || trainData.labels.empty()) {
        std::cerr << "Dataset is empty. Please check the data file and path." << std::endl;
        return -1;
    }

    // try {
    //     int index = 0;
    //     //std::vector<float> image = trainData.getImageByIndex(index);

    //     // Print the label and the image
    //     std::cout << "Label of the digit at index " << index << ": " << trainData.labels[index] << std::endl;
    //     std::cout << "Digit image at index " << index << ":" << std::endl;
    //     printDigitImage(trainData.images[index]);
    // } catch (const std::out_of_range& e) {
    //     std::cerr << "Error: " << e.what() << std::endl;
    // }

    int epochs = 50;
    int batchSize = 64;
    float learningRate = 0.001;
    trainModel(trainData, epochs, batchSize, learningRate);


    return 0;
}
#endif