#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"

#include "flow-code/imageLib/Image.h"
#include "flow-code/flowIO.h"

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
 * @param ocl_input cl_mem input object
 * @param ocl_output cl_mem output object
 * @param width input matrix width
 * @param height input matrix height
 * @param downsample_rate how much we're downsampling by
 */
void Convolve(
    cl_command_queue queue, 
    cl_kernel kernel, 
    cl_uint work_dim, 
    const size_t *local_work_size, 
    cl_mem *ocl_input, 
    cl_mem *ocl_output,
    cl_mem *ocl_weights,
    size_t width,
    size_t height,
    size_t filter_radius,
    size_t stride
) 
{
    cl_int err;
    size_t ds_width = width / stride;
    size_t ds_height = height / stride;

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
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), ocl_weights);
    CHECK_ERR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 3, sizeof(int), &filter_radius);
    CHECK_ERR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 4, sizeof(int), &stride);
    CHECK_ERR(err, "clSetKernelArg");
    err |= clSetKernelArg(kernel, 5, sizeof(int), &width);
    CHECK_ERR(err, "clSetKernelArg");
    err |= clSetKernelArg(kernel, 6, sizeof(int), &height);
    CHECK_ERR(err, "clSetKernelArg");
    err |= clSetKernelArg(kernel, 7, sizeof(int), &ds_width);
    CHECK_ERR(err, "clSetKernelArg");
    err |= clSetKernelArg(kernel, 8, sizeof(int), &ds_height);
    CHECK_ERR(err, "clSetKernelArg");

    // Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
}

void CalculateTemporalGradient(
    cl_command_queue queue, 
    cl_kernel kernel, 
    cl_uint work_dim, 
    const size_t *local_work_size, 
    cl_mem *ocl_input1, 
    cl_mem *ocl_input2,
    cl_mem *ocl_output,
    size_t width,
    size_t height
) 
{
    cl_int err;

    // round to make sure GWS % LWS == 0
    size_t global_work_size[2] = {
        ((width + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0],
        ((height + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1]
    };

    // Kernel arguments

    //__global const float* src, __global float* dst, const int src_width, 
    // const int src_height
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), ocl_input1);
    CHECK_ERR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), ocl_input2);
    CHECK_ERR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), ocl_output);
    CHECK_ERR(err, "clSetKernelArg");
    err |= clSetKernelArg(kernel, 3, sizeof(int), &width);
    CHECK_ERR(err, "clSetKernelArg");
    err |= clSetKernelArg(kernel, 4, sizeof(int), &height);
    CHECK_ERR(err, "clSetKernelArg");

    // Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
}

void OpenCLOpticalFlow(Matrix *input0, Matrix *input1, Matrix *result)
{
    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel("opticalFlow.cl"); // Load kernel source

    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel convolution_kernel;          // kernel
    cl_kernel temporal_gradient_kernel;          // kernel
    cl_kernel solver_kernel;          // kernel

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    //@@ define local and global work sizes
    size_t local_item_size[2] = {16, 16};
    size_t global_item_size[2];


    float gaussian_weights[25] = {
        0.00390625f, 0.015625f, 0.0234375f, 0.015625f, 0.00390625f,
        0.015625f,   0.0625f,   0.09375f,   0.0625f,   0.015625f, 
        0.0234375f,  0.09375f,  0.140625f,  0.09375f,  0.0234375f,
        0.015625f,   0.0625f,   0.09375f,   0.0625f,   0.015625f,
        0.00390625f, 0.015625f, 0.0234375f, 0.015625f, 0.00390625f
    };
    size_t gaussian_radius = 2;

    float sobel_x_weights[9] = {
        -1.0f,  0.0f,  1.0f,
        -2.0f,  0.0f,  2.0f,
        -1.0f,  0.0f,  1.0f
    };
    float sobel_y_weights[9] = {
        -1.0f, -2.0f, -1.0f,
        0.0f,  0.0f,  0.0f,
        1.0f,  2.0f,  1.0f
    };
    size_t sobel_radius = 1;

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
    convolution_kernel = clCreateKernel(program, "conv2d", &err);
    CHECK_ERR(err, "clCreateKernel");
    temporal_gradient_kernel = clCreateKernel(program, "temporalGradient", &err);
    CHECK_ERR(err, "clCreateKernel");

    printf("Kernels created\n");
    
    // Allocate GPU here
    int height = input0->shape[0];
    int width = input0->shape[1];

    // stores frames on a pyramid layer basis
    cl_mem frame1s[PYR_LAYERS];
    cl_mem frame2s[PYR_LAYERS];
    cl_mem frame1_Ix[PYR_LAYERS];
    cl_mem frame1_Iy[PYR_LAYERS];
    cl_mem frame2_Ix[PYR_LAYERS];
    cl_mem frame2_Iy[PYR_LAYERS];
    cl_mem It[PYR_LAYERS];

    // stores convolution filters
    cl_mem gaussian_2d;
    cl_mem sobel_x;
    cl_mem sobel_y;

    // allocate storage for pyramid layers
    for (int i = 0; i < PYR_LAYERS; ++i) {
        frame1s[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
        CHECK_ERR(err, "clCreateBuffer");
        frame2s[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
        CHECK_ERR(err, "clCreateBuffer");
        frame1_Ix[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
        CHECK_ERR(err, "clCreateBuffer");
        frame1_Iy[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
        CHECK_ERR(err, "clCreateBuffer");
        frame2_Ix[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
        CHECK_ERR(err, "clCreateBuffer");
        frame2_Iy[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
        CHECK_ERR(err, "clCreateBuffer");
        It[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
        CHECK_ERR(err, "clCreateBuffer");
    }

    // allocate storage for convolution kernels
    gaussian_2d = clCreateBuffer(context, CL_MEM_READ_ONLY, (gaussian_radius * 2 + 1) * (gaussian_radius * 2 + 1) * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer");
    sobel_x = clCreateBuffer(context, CL_MEM_READ_ONLY, (sobel_radius * 2 + 1) * (sobel_radius * 2 + 1) * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer");
    sobel_y = clCreateBuffer(context, CL_MEM_READ_ONLY, (sobel_radius * 2 + 1) * (sobel_radius * 2 + 1) * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer");

    // enqueue initial images
    err = clEnqueueWriteBuffer(queue, frame1s[0], CL_TRUE, 0, height * width * sizeof(float), input0->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer");
    err = clEnqueueWriteBuffer(queue, frame2s[0], CL_TRUE, 0, height * width * sizeof(float), input1->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer");

    // enqueue convolution kernels
    err = clEnqueueWriteBuffer(queue, gaussian_2d, CL_TRUE, 0, (gaussian_radius * 2 + 1) * (gaussian_radius * 2 + 1) * sizeof(float), gaussian_weights, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer");
    err = clEnqueueWriteBuffer(queue, sobel_x, CL_TRUE, 0, (sobel_radius * 2 + 1) * (sobel_radius * 2 + 1) * sizeof(float), sobel_x_weights, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer");
    err = clEnqueueWriteBuffer(queue, sobel_y, CL_TRUE, 0, (sobel_radius * 2 + 1) * (sobel_radius * 2 + 1) * sizeof(float), sobel_y_weights, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer");

    // downsample each layer into the next
    for (int i = 0; i < PYR_LAYERS - 1; ++i) {
        Convolve(
            queue, convolution_kernel, WORK_DIM, local_item_size, &frame1s[i], &frame1s[i+1], &gaussian_2d, width / pow(DSR, i), height / pow(DSR, i), gaussian_radius, DSR
        );
        Convolve(
            queue, convolution_kernel, WORK_DIM, local_item_size, &frame2s[i], &frame2s[i+1], &gaussian_2d, width / pow(DSR, i), height / pow(DSR , i), gaussian_radius, DSR
        );
    }

    // calculate spatial gradients
    for (int i = 0; i < PYR_LAYERS; ++i) {
        Convolve(
            queue, convolution_kernel, WORK_DIM, local_item_size, &frame1s[i], &frame1_Ix[i], &sobel_x, width / pow(DSR, i), height / pow(DSR, i), sobel_radius, 1
        );
        Convolve(
            queue, convolution_kernel, WORK_DIM, local_item_size, &frame1s[i], &frame1_Iy[i], &sobel_y, width / pow(DSR, i), height / pow(DSR, i), sobel_radius, 1
        );
        Convolve(
            queue, convolution_kernel, WORK_DIM, local_item_size, &frame2s[i], &frame2_Ix[i], &sobel_x, width / pow(DSR, i), height / pow(DSR, i), sobel_radius, 1
        );
        Convolve(
            queue, convolution_kernel, WORK_DIM, local_item_size, &frame2s[i], &frame2_Iy[i], &sobel_y, width / pow(DSR, i), height / pow(DSR, i), sobel_radius, 1
        );
    }

    // calculate temporal gradient
    for (int i = 0; i < PYR_LAYERS; i++) {
        CalculateTemporalGradient(
            queue, temporal_gradient_kernel, WORK_DIM, local_item_size, &frame1s[i], &frame2s[i], &It[i], width / pow(DSR, i), height / pow(DSR, i)
        );
    }

    

    //@@ Copy the GPU memory back to the CPU here
    /*err = clEnqueueReadBuffer(queue, It[PYR_LAYERS - 1], CL_TRUE, 0, height * width / pow(pow(DSR, PYR_LAYERS - 1), 2) * sizeof(float), result->data, 0, NULL, NULL);
    result->shape[0] = height / pow(DSR, PYR_LAYERS - 1);
    result->shape[1] = width / pow(DSR, PYR_LAYERS - 1);*/

    err = clEnqueueReadBuffer(queue, It[0], CL_TRUE, 0, height * width * sizeof(float), result->data, 0, NULL, NULL);
    result->shape[0] = height;
    result->shape[1] = width;
    CHECK_ERR(err, "clEnqueueReadBuffer");


    //@@ Free the GPU memory here
    for (int i = 0; i < PYR_LAYERS; i++) {
        clReleaseMemObject(frame1s[i]);
        clReleaseMemObject(frame2s[i]);
        clReleaseMemObject(frame1_Ix[i]);
        clReleaseMemObject(frame2_Ix[i]);
        clReleaseMemObject(frame1_Iy[i]);
        clReleaseMemObject(frame2_Iy[i]);
        clReleaseMemObject(It[i]);
    }

    clReleaseProgram(program);
    clReleaseKernel(convolution_kernel);
    clReleaseKernel(temporal_gradient_kernel);
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
    int output_width = input_width; //(((input_width + 1) / 2) + 1) / 2;
    int output_height = input_height; //(((input_height + 1) / 2) + 1) / 2;

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

    CShape shape(input_width, input_height, 2);
    CFloatImage img(shape);
    for (int x = 0; x < input_width; x++) {
        for (int y = 0; y < input_height; y++) {
            img.Pixel(x, y, 0) = host_c.data[y * input_width + x];
            img.Pixel(x, y, 1) = host_c.data[y * input_width + x];
        }
    }
    WriteFlowFile(img, "output.png");

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
