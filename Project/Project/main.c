#include <stdio.h>
#include <stdlib.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void OpenCLOpticalFlow(Matrix *input0, Matrix *result)
{
    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel("opticalFlow.cl"); // Load kernel source

    // Device input and output buffers
    // frame 1, output for now
    cl_mem device_a, device_c;

    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get the ID for the specified kind of device type.
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
# if __APPLE__
    queue = clCreateCommandQueue(context, device_id, 0, &err);
#else
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
#endif
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "opticalFlow", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Allocate GPU memory here
    device_a = clCreateBuffer(context, CL_MEM_READ_ONLY, (*input0).shape[0] * (*input0).shape[1] * sizeof(unsigned int), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer1");
    device_c = clCreateBuffer(context, CL_MEM_READ_ONLY, (*result).shape[0] * (*result).shape[1] * sizeof(unsigned int), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer3");


    //@@ Copy memory to the GPU here
    err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, (*input0).shape[0] * (*input0).shape[1] * sizeof(unsigned int), (*input0).data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer1");


    //@@ define local and global work sizes
    unsigned int size_a = (*input0).shape[0] * (*input0).shape[1];
    size_t local_item_size[2] = {16, 16};
    //err = clGetDeviceInfo(*device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_item_size, NULL);
    size_t global_item_size[2];

    // across rows first, col second
    global_item_size[0] = ((input0->shape[1] + local_item_size[0] - 1) / local_item_size[0]) * local_item_size[0];
    global_item_size[1] = ((input0->shape[0] + local_item_size[1] - 1) / local_item_size[1]) * local_item_size[1];




    // Set the arguments to our compute kernel
    // __global const int *A, __global const int *B, __global int *C,
    // const unsigned int numARows, const unsigned int numAColumns,

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_a);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_c);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 2, sizeof(int), &input0->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(kernel, 3, sizeof(int), &input0->shape[1]);

    //@@ Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");

    //@@ Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, size_a * sizeof(int), (*result).data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer");

    //@@ Free the GPU memory here
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernel_source);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    int input_width, input_height, input_channels;

    // load frame 1
    unsigned char *temp_pixels = stbi_load(input_file_a, &input_width, &input_height, &input_channels, 1);
    host_a.shape[0] = input_height;
    host_a.shape[1] = input_width;
    host_a.data = (int *)malloc(sizeof(int) * input_height * input_width);
    for (int i = 0; i < input_height * input_width; i++) {
        host_a.data[i] = temp_pixels[i]; 
    }

    // load frame 2
    temp_pixels = stbi_load(input_file_b, &input_width, &input_height, &input_channels, 1);
    host_b.shape[0] = input_height;
    host_b.shape[1] = input_width;
    host_b.data = (int *)malloc(sizeof(int) * input_height * input_width);
    for (int i = 0; i < input_height * input_width; i++) {
        host_b.data[i] = temp_pixels[i]; 
    }

    // Allocate the memory for the target.
    host_c.shape[0] = input_height;
    host_c.shape[1] = input_width;
    host_c.data = (int *)malloc(sizeof(int) * host_c.shape[0] * host_c.shape[1]);

    // Call your optical flow.
    OpenCLOpticalFlow(&host_a, &host_c);

    unsigned char *output_bytes = (unsigned char *)malloc(input_height * input_width);
    for (int i = 0; i < input_height * input_width; i++) {
        output_bytes[i] = (unsigned char)host_c.data[i];
    }

    int output_channels = 1;
    stbi_write_png(input_file_d, input_width, input_height, output_channels, output_bytes, input_width * output_channels);



    // Check the result of the matrix multiply
    //CheckMatrix(&answer, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    //free(answer.data);

    return 0;
}
