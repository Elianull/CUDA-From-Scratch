#include <cmath>
#include <iostream>
#include <cstdlib>

__global__ void crossEntropyLossKernel(float* predictions, float* targets, float* loss, int numClasses, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;
    
    float sampleLoss = 0.0f;
    for (int c = 0; c < numClasses; ++c) {
        float p = predictions[idx * numClasses + c];
        float t = targets[idx * numClasses + c];
        sampleLoss -= t * logf(p + 1e-7f);
    }

    atomicAdd(loss, sampleLoss);
}

float crossEntropyLoss(float* predictions, float* targets, int numClasses, int totalSize, int batchSize) {
    float* d_loss;
    float totalLoss = 0.0f;
    cudaMalloc(&d_loss, sizeof(float));

    for (int start = 0; start < totalSize; start += batchSize) {
        int currentBatchSize = min(batchSize, totalSize - start);

        cudaMemset(d_loss, 0, sizeof(float));

        int blockSize = 256;
        int numBlocks = (currentBatchSize + blockSize - 1) / blockSize;
        crossEntropyLossKernel<<<numBlocks, blockSize>>>(predictions + start * numClasses, 
                                                         targets + start * numClasses, 
                                                         d_loss, numClasses, currentBatchSize);

        float batchLoss;
        cudaMemcpy(&batchLoss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        totalLoss += batchLoss; // Accumulate loss from each batch
    }

    cudaFree(d_loss);

    return totalLoss / totalSize; // Return the average loss over all samples
}

#ifdef COMPILE_MAIN
int main(int argc, char* argv[]) {
    srand(time(0)); // Seed the random number generator

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <numClasses> <totalSize> <batchSize>\n";
        return 1;
    }

    int numClasses = std::atoi(argv[1]);
    int totalSize = std::atoi(argv[2]); // Total size of the dataset
    int batchSize = std::atoi(argv[3]);
    
    // Adjustments for allocating and initializing predictions and targets
    float* h_predictions = new float[totalSize * numClasses];
    float* h_targets = new float[totalSize * numClasses]; // For simplicity, targets are also floats
    
    // Generate random predictions and one-hot encoded targets
    for (int i = 0; i < totalSize; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < numClasses; ++j) {
            h_predictions[i * numClasses + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            sum += h_predictions[i * numClasses + j];
        }
        // Normalize predictions to satisfy the simplex constraint
        for (int j = 0; j < numClasses; ++j) {
            h_predictions[i * numClasses + j] /= sum;
        }
        
        // Generate a random target class (one-hot encoding for simplicity)
        int targetClass = rand() % numClasses;
        for (int j = 0; j < numClasses; ++j) {
            h_targets[i * numClasses + j] = (j == targetClass) ? 1.0f : 0.0f;
        }
    }

    float* d_predictions;
    float* d_targets;
    cudaMalloc(&d_predictions, totalSize * numClasses * sizeof(float));
    cudaMalloc(&d_targets, totalSize * numClasses * sizeof(float));

    cudaMemcpy(d_predictions, h_predictions, totalSize * numClasses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, totalSize * numClasses * sizeof(float), cudaMemcpyHostToDevice);

    float loss = crossEntropyLoss(d_predictions, d_targets, numClasses, totalSize, batchSize);

    std::cout << "Average Cross-Entropy Loss: " << loss << std::endl;

    // Free host and device memory
    delete[] h_predictions;
    delete[] h_targets;
    cudaFree(d_predictions);
    cudaFree(d_targets);

    return 0;
}
#endif
