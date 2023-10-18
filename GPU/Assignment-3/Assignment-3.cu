#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

const int N = 512;

__global__ void matrixTranspose(float* A, float* B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    B[col * N + row] = A[row * N + col];
}

__global__ void matrixMultiply(float* A, float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;

    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

int main() {
    float* h_A, * h_B, * h_C;
    float* d_A, * d_B, * d_C;

    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];

    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(N / 16, N / 16);

    matrixMultiply << <gridSize, blockSize >> > (d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}