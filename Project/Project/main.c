#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

// Local work size constant
#define LWS 16

// Downsampling rate
#define DSR 2

// Work Dimensionality
#define WORK_DIM 2

// Number of Pyramid Layers
#define PYR_LAYERS 3

/**
 * Interface to easily downsample a single frame 
 * 
 * @param queue cl_command_queue
 * @param kernel cl_kernel
 * @param work_dim work dimensionality
 * @param local_work_size can calculate GWS off this and height/width
 * @param r_input raw float array input
 * @param ocl_input cl_mem input object
 * @param ocl_output cl_mem output object
 * @param width input matrix width
 * @param height input matrix height
 * @param downsample_rate how much we're downsampling by
 */
void Downsample(
    cl_command_queue queue, 
    cl_kernel kernel, 
    cl_uint work_dim, 
    const size_t *local_work_size, 
    Matrix *r_input,
    cl_mem *ocl_input, 
    cl_mem *ocl_output,
    size_t width,
    size_t height,
    size_t downsample_rate
) 
{
    cl_int err;
    size_t ds_width = width / downsample_rate;
    size_t ds_height = width / downsample_rate;

    // round to make sure GWS % LWS == 0
    size_t global_work_size[2] = {
        ((ds_width + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0],
        ((ds_height + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1]
    };

    // Kernel arguments

    //__global const float* src, __global float* dst, const int src_width, 
    // const int src_height, const int dst_width, const int dst_height
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), ocl_input);
    CHECK_ERR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), ocl_output);
    CHECK_ERR(err, "clSetKernelArg");
    err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    CHECK_ERR(err, "clSetKernelArg");
    err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    CHECK_ERR(err, "clSetKernelArg");
    err |= clSetKernelArg(kernel, 4, sizeof(int), &ds_width);
    CHECK_ERR(err, "clSetKernelArg");
    err |= clSetKernelArg(kernel, 5, sizeof(int), &ds_height);
    CHECK_ERR(err, "clSetKernelArg");

    // Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
}

void OpenCLOpticalFlow(Matrix *input0, Matrix *input1, Matrix *result)
{
    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel("opticalFlow.cl"); // Load kernel source

    // Device input and output buffers (float, 0 to 1.0f)
    // device_a0 = frame 1 level 0 (no downsample)
    // device_b0 = frame 2 
    // device_a1 = frame 1 level 1 (downsampled with gaussian blur 1/2)
    // device_b1 = frame 2 
    // device_a2 = frame 1 level 2 (downsampled with gaussian blur 1/4)
    // device_b2 = frame 2
    // device_
    // device_c = output (float, 0 to 1.0f)
    cl_mem device_a0, device_b0, device_a1, device_b1, device_a2, device_b2, device_c;

    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel downsample_kernel;          // kernel
    cl_kernel gradient_kernel;          // kernel
    cl_kernel solver_kernel;          // kernel

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    //@@ define local and global work sizes
    size_t local_item_size[2] = {16, 16};
    size_t global_item_size[2];


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
    downsample_kernel = clCreateKernel(program, "lk_downsample", &err);
    CHECK_ERR(err, "clCreateKernel");

    printf("Kernels created\n");
    
    // Allocate GPU here
    int height = input0->shape[0];
    int width = input0->shape[1];

    // Output
    device_c = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer");

    // stores frames on a pyramid layer basis
    cl_mem frame1s[PYR_LAYERS];
    cl_mem frame2s[PYR_LAYERS];

    // allocate storage for pyramid layers
    for (int i = 0; i < PYR_LAYERS; ++i) {
        frame1s[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
        CHECK_ERR(err, "clCreateBuffer");
        frame2s[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
        CHECK_ERR(err, "clCreateBuffer");
    }

    // enqueue initial images
    err = clEnqueueWriteBuffer(queue, frame1s[0], CL_TRUE, 0, height * width * sizeof(float), input0->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer");
    err = clEnqueueWriteBuffer(queue, frame2s[0], CL_TRUE, 0, height * width * sizeof(float), input1->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer");

    // downsample each layer into the next
    for (int i = 0; i < 2; ++i) {
        Downsample(
            queue, downsample_kernel, WORK_DIM, local_item_size, input0, &frame1s[i], &frame1s[i+1], width / pow(DSR, i), height / pow(DSR, i), DSR
        );
        Downsample(
            queue, downsample_kernel, WORK_DIM, local_item_size, input1, &frame2s[i], &frame2s[i+1], width / pow(DSR, i), height / pow(DSR , i), DSR
        );
    }

    //@@ Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue, frame1s[PYR_LAYERS - 1], CL_TRUE, 0, height * width / pow(pow(DSR, PYR_LAYERS - 1), 2) * sizeof(float), result->data, 0, NULL, NULL);
    result->shape[0] = height / pow(DSR, PYR_LAYERS - 1);
    result->shape[1] = width / pow(DSR, PYR_LAYERS - 1);
    CHECK_ERR(err, "clEnqueueReadBuffer");


    //@@ Free the GPU memory here
    for (int i = 0; i < PYR_LAYERS; i++) {
        clReleaseMemObject(frame1s[i]);
        clReleaseMemObject(frame2s[i]);
    }

    clReleaseMemObject(device_c);
    clReleaseProgram(program);
    clReleaseKernel(downsample_kernel);
    //clReleaseKernel(gradient_kernel);
    //clReleaseKernel(solver_kernel);
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
    // host_a = input frame 1
    // host_b = input frame 2
    // host_c = output frame
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    int input_width, input_height, input_channels;

    // load frame 1, cast to float
    printf("Load frame 1...\n");
    unsigned char *temp_pixels = stbi_load(input_file_a, &input_width, &input_height, &input_channels, 1);
    host_a.shape[0] = input_height;
    host_a.shape[1] = input_width;
    host_a.data = (float *)malloc(sizeof(float) * input_height * input_width);
    for (int i = 0; i < input_height * input_width; i++) {
        host_a.data[i] = (float)((float)temp_pixels[i])/255.0f; 
    }

    // load frame 2, cast to float
    printf("Load frame 2...\n");
    temp_pixels = stbi_load(input_file_b, &input_width, &input_height, &input_channels, 1);
    host_b.shape[0] = input_height;
    host_b.shape[1] = input_width;
    host_b.data = (float *)malloc(sizeof(float) * input_height * input_width);
    for (int i = 0; i < input_height * input_width; i++) {
        host_b.data[i] = (float)((float)temp_pixels[i])/255.0f; 
    }

    // Allocate the memory for the output.
    printf("Allocate output...\n");
    int output_channels = 1;
    int output_width = (((input_width + 1) / 2) + 1) / 2;
    int output_height = (((input_height + 1) / 2) + 1) / 2;

    host_c.shape[0] = output_height;
    host_c.shape[1] = output_width;
    host_c.data = (float *)malloc(sizeof(float) * host_c.shape[0] * host_c.shape[1]);

    // Call your optical flow.
    printf("Start optical flow...\n");
    OpenCLOpticalFlow(&host_a, &host_b, &host_c);

    printf("Read output...\n");
    unsigned char *output_bytes = (unsigned char *)malloc(output_height * output_width);
    for (int i = 0; i < output_height * output_width; i++) {
        output_bytes[i] = (unsigned char)(host_c.data[i] * 255);
    }

    stbi_write_png(input_file_d, output_width, output_height, output_channels, output_bytes, output_width * output_channels);
    SaveMatrix("output.raw", &host_c);



    // Check the result of the matrix multiply
    //CheckMatrix(&answer, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    //free(answer.data);
    printf("Done.\n");
    return 0;
}
