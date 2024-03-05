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
        //printf("updateWeightsAndBiasesKernel: weights[%d] = %f, biases[%d] = %f\n", idx, weights[idx], col, biases[col]); // Debugging statement
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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxGridSize = prop.maxGridSize[0];
    int maxBlockSize = prop.maxThreadsPerBlock;
    printf("Max grid size: %d\n", maxGridSize);
    printf("Max block size: %d\n", maxBlockSize);

    int inputSize = 784; // 28*28 images
    int numOutput = 10; // 10 classes for MNIST digits
    int imageSize = 28 * 28; // Size of each image

    std::vector<float> flattenedImages;
    for (const auto& image : trainData.images) {
        flattenedImages.insert(flattenedImages.end(), image.begin(), image.end());
    }

    float *d_weights, *d_biases, *d_output, *d_images, *d_targets;
    float *d_gradWeights, *d_gradBiases, *d_gradOutput;
    int *d_labels;
    cudaMalloc(&d_weights, inputSize * numOutput * sizeof(float));
    cudaCheckError();
    cudaMalloc(&d_biases, numOutput * sizeof(float));
    cudaCheckError();
    cudaMalloc(&d_output, batchSize * numOutput * sizeof(float));
    cudaCheckError();
    cudaMalloc(&d_images, flattenedImages.size() * sizeof(float));
    cudaCheckError();
    cudaMalloc(&d_labels, trainData.labels.size() * sizeof(int));
    cudaCheckError();
    cudaMalloc(&d_targets, batchSize * numOutput * sizeof(float));
    cudaCheckError();
    cudaMalloc(&d_gradWeights, inputSize * numOutput * sizeof(float));
    cudaCheckError();
    cudaMalloc(&d_gradBiases, numOutput * sizeof(float));
    cudaCheckError();
    cudaMalloc(&d_gradOutput, batchSize * numOutput * sizeof(float));
    cudaCheckError();

    cudaMemcpy(d_images, flattenedImages.data(), flattenedImages.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    initializeWeightsAndBiases(d_weights, d_biases, inputSize, numOutput);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0;
        int numBatches = trainData.images.size() / batchSize;

        for (int batch = 0; batch < numBatches; ++batch) {
            int offset = batch * batchSize;

            // Copy batch data to device memory
            cudaMemcpy(d_images + offset * imageSize, &flattenedImages[offset * imageSize], batchSize * imageSize * sizeof(float), cudaMemcpyHostToDevice);
            cudaCheckError();
            cudaMemcpy(d_labels, &trainData.labels[offset], batchSize * sizeof(int), cudaMemcpyHostToDevice);
            cudaCheckError();

            // Forward pass
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((batchSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (numOutput + threadsPerBlock.y - 1) / threadsPerBlock.y);
            //printf("Forward pass - numBlocks: (%d, %d), threadsPerBlock: (%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
            denseLayerKernel<<<numBlocks, threadsPerBlock>>>(d_images + offset * imageSize, d_weights, d_biases, d_output, batchSize, inputSize, numOutput);
            cudaCheckError();
            softmax(d_output, d_output, batchSize * numOutput);
            cudaCheckError();

            convertLabelsToOneHot<<<(batchSize + 255) / 256, 256>>>(d_labels, d_targets, batchSize, numOutput);
            cudaCheckError();

            // Compute loss
            float batchLoss = crossEntropyLoss(d_output, d_targets, numOutput, batchSize, batchSize);
            cudaCheckError();
            totalLoss += batchLoss;
            // if (epoch == 0) {
            //     printf("Batch %d: Loss = %f\n", batch, batchLoss); // Debugging statement
            // }

            // Backward pass
            softmaxGradKernel<<<numBlocks, threadsPerBlock>>>(d_output, d_targets, d_gradOutput, batchSize, numOutput);
            cudaCheckError("softmaxGradKernel");
            // printf("numBlocks: (%d, %d, %d)\n", numBlocks.x, numBlocks.y, numBlocks.z);
            // printf("threadsPerBlock: (%d, %d, %d)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
            // printf("Shared memory size: %zu bytes\n", (inputSize + numOutput) * threadsPerBlock.x * threadsPerBlock.y * sizeof(float));
            // printf("d_images: %p, offset: %d, imageSize: %d\n", d_images, offset, imageSize);
            // printf("d_gradOutput: %p\n", d_gradOutput);
            // printf("d_gradWeights: %p\n", d_gradWeights);
            // printf("batchSize: %d, inputSize: %d, numOutput: %d\n", batchSize, inputSize, numOutput);

            int maxSharedMemPerBlock = 49152; // 48 KB in bytes
            int chunkSize = (inputSize + threadsPerBlock.x - 1) / threadsPerBlock.x;
            //int numChunks = (inputSize + chunkSize - 1) / chunkSize;
            int sharedMemSize = min(maxSharedMemPerBlock, static_cast<int>((chunkSize + numOutput) * threadsPerBlock.x * sizeof(float)));

            denseLayerGradientKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_images + offset * imageSize,
                d_gradOutput,
                d_gradWeights,
                batchSize,
                inputSize,
                numOutput
            );
            cudaCheckError("denseLayerGradientKernel");
            
            biasGradientKernel<<<numBlocks.x, numOutput>>>(d_gradOutput, d_gradBiases, batchSize, numOutput);
            cudaCheckError("biasGradientKernel");

            // Weight and bias update
            dim3 weightsBlocks((numOutput + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (inputSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
            updateWeightsAndBiasesKernel<<<weightsBlocks, threadsPerBlock>>>(d_weights, d_biases, d_gradWeights, d_gradBiases, inputSize, numOutput, learningRate);
            cudaCheckError();
        }

        std::cout << "Epoch " << epoch << ": Average Loss = " << totalLoss / numBatches << std::endl;
    }

    // Cleanup: Free device memory
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_targets);
    cudaFree(d_gradWeights);
    cudaFree(d_gradBiases);
    cudaFree(d_gradOutput);
}

#ifdef COMPILE_MAIN
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

    int epochs = 100;
    int batchSize = 64;
    float learningRate = 0.001;
    trainModel(trainData, epochs, batchSize, learningRate);


    return 0;
}
#endif
