// Bitonic Sort Parallel Implementation in cuda

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

// Bitonic Sort Kernel
__device__ void bitonicMerge(int* dev_values, int j, int k, int size, int dir) {
    int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;
    if (i < size && ixj > i) {
        if ((ixj & k) == 0) {
            if ((dev_values[i] > dev_values[ixj]) == dir) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        else {
            if ((dev_values[i] < dev_values[ixj]) == dir) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

__global__ void bitonicSort(int* dev_values, int size, int dir) {
    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            bitonicMerge(dev_values, j, k, size, dir);
            __syncthreads();
        }
    }
}

int main() {
    const int size = 1024;
    int values[size];

    // Initialize the array with random values
    for (int i = 0; i < size; ++i) {
        values[i] = rand() % 1000;
    }

    // Display original array
    std::cout << "Original Array:\n";
    for (int i = 0; i < size; ++i) {
        std::cout << values[i] << " ";
    }
    std::cout << "\n\n";

    // Allocate device memory and copy data to device
    int* dev_values;
    cudaMalloc((void**)&dev_values, size * sizeof(int));
    cudaMemcpy(dev_values, values, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    bitonicSort<<<dimGrid, dimBlock>>>(dev_values, size, 1);  // 1 for ascending order

    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(values, dev_values, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Display sorted array
    std::cout << "Sorted Array:\n";
    for (int i = 0; i < size; ++i) {
        std::cout << values[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Execution Time for Bitonic Sort: " << milliseconds << " ms\n"; 
    // Clean up
    cudaFree(dev_values);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
