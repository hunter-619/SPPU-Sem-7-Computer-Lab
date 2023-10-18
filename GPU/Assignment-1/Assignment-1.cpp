#define CL_TARGET_OPENCL_VERSION 300

#include <CL/Opencl.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// OpenCL kernel for matrix multiplication
const char* kernelSource = R"(
    __kernel void matrixMultiply(__global float* A,
                                __global float* B,
                                __global float* C,
                                const int M,
                                const int N,
                                const int K) {
        int row = get_global_id(0);
        int col = get_global_id(1);

        float sum = 0.0f;
        for (int x = 0; x < K; x++) {
            sum += A[row * K + x] * B[x * N + col];
        }

        C[row * N + col] = sum;
    }
)";

void matrixMultiply(vector<vector<float>> &A, 
                    vector<vector<float>> &B,
                    vector<vector<float>> &C,
                    const int M,
                    const int N,
                    const int K) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < K; j++) {
            for(int k = 0; k < M; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


int main() {
    // OpenCL implementation of matrix multiplication

    // Matrix dimensions
    const int M = 3 ;
    const int N = 3;
    const int K = 3;

    // Create matrices A, B, and C
    float A[M * K] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[K * N] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    float C[M * N] = {0};

    // Load OpenCL platform(drivers) and device
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create OpenCL context and command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Create OpenCL buffers for matrices A, B, and C
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * K, A, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * K * N, B, NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, NULL);

    // Build and compile the OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    cl_event startEvent, endEvent;
    clEnqueueMarkerWithWaitList(queue, 0, NULL, &startEvent);

    // Execute the kernel
    size_t globalSize[2] = {M, N};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    // clFinish(queue);

    clEnqueueMarkerWithWaitList(queue, 0, NULL, &endEvent);
    clFinish(queue);

    // Read the result back from the device
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * M * N, C, 0, NULL, NULL);

    // Calculate execution time
    cl_ulong startTime, endTime;
    clGetEventProfilingInfo(startEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
    clGetEventProfilingInfo(endEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);

    // Calculate execution time in nanoseconds
    cl_ulong executionTime = endTime - startTime;

    // Convert to milliseconds for readability
    // double executionTimeNS = (double)executionTime;

    printf("OpenCL Execution time is: %lld nanoseconds \n", executionTime);

    // Print the result matrix
    printf("Result Matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f\t", C[i * N + j]);
        }
        printf("\n");
    }

    // Clean up resources
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseEvent(startEvent);
    clReleaseEvent(endEvent);

    printf("\n");


    // CPP implementation of matrix multiplication
    // Initialize matrices
    vector<vector<float>> A1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    vector<vector<float>> B1 = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};;
    vector<vector<float>> C1(N, vector<float>(M, 0));

    auto start = high_resolution_clock::now();
    matrixMultiply(A1, B1, C1, N, M, K);
    auto stop = high_resolution_clock:: now();
    
    auto duration = duration_cast<nanoseconds>(stop - start);
    printf("CPP Execution time is: %lld nanoseconds \n", duration.count());

    printf("Result Matrix C1:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f\t", C1[i][j]);
        }
        printf("\n");
    }

    return 0;
}
