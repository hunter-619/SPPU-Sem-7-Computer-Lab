#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

const int N = 100000;

__global__ void dotProduct(float* a, float* b, float* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    float tempResult = 0.0f;
    if (idx < N) {
        tempResult += a[idx] * b[idx];
    }

    atomicAdd(result, tempResult);
}

int main() {
    float* h_a, * h_b, * d_a, * d_b, * d_result;
    float result = 0.0f;

    h_a = new float[N];
    h_b = new float[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = 1;
        h_b[i] = 1;
    }

    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    int numThreadsPerBlock = 256;
    int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Create CUDA events for measuring time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    dotProduct << <numBlocks, numThreadsPerBlock >> > (d_a, d_b, d_result);

    // Record the stop event
    cudaEventRecord(stop);

    // Synchronize to make sure the stop event finishes
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result back from the device to the host
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Output results
    std::cout << "Dot Product: " << result << std::endl;
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    // Clean Memory
    delete[] h_a;
    delete[] h_b;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}