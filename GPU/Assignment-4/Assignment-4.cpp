#define CL_TARGET_OPENCL_VERSION 300

#include <CL/opencl.h>
#include <CL/cl.hpp>
#include <iostream>

const char* kernelSource = R"(
    __kernel void helloWorld() { 
        printf("Hello, World!\n"); 
    }
)";

int main() {
    // Get the platform(drivers)
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // Get the device
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // Create the context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

    // Create the command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // Load and compile the OpenCL kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "helloWorld", nullptr);

    // Execute the OpenCL kernel
    size_t globalWorkSize = 1;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
