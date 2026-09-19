#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstdlib> 
#include <clblast.h>
#include <vector>

#include "device.h"
#include "kernel.h"
#include "matrix.h"

#include "flow-code/imageLib/Image.h"
#include "flow-code/flowIO.h"
#include "flow-code/colorcode.h"

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

const float gaussian_weights[25] = {
    0.00390625f, 0.015625f, 0.0234375f, 0.015625f, 0.00390625f,
    0.015625f,   0.0625f,   0.09375f,   0.0625f,   0.015625f, 
    0.0234375f,  0.09375f,  0.140625f,  0.09375f,  0.0234375f,
    0.015625f,   0.0625f,   0.09375f,   0.0625f,   0.015625f,
    0.00390625f, 0.015625f, 0.0234375f, 0.015625f, 0.00390625f
};
const size_t gaussian_radius = 2;

const float sobel_x_weights[9] = {
    -1.0f,  0.0f,  1.0f,
    -2.0f,  0.0f,  2.0f,
    -1.0f,  0.0f,  1.0f
};
const float sobel_y_weights[9] = {
    -1.0f, -2.0f, -1.0f,
    0.0f,  0.0f,  0.0f,
    1.0f,  2.0f,  1.0f
};
const size_t sobel_radius = 1;

/**
 * Interface to easily downsample a single frame 
 * 
 * @param queue cl_command_queue
 * @param kernel cl_kernel
 * @param work_dim work dimensionality
 * @param local_work_size can calculate GWS off this and height/width
 * @param ocl_input cl_mem input object
 * @param ocl_output cl_mem output object
 * @param ocl_weights filter weights in kernel
 * @param width input matrix width
 * @param height input matrix height
 * @param filter_radius radius of filter
 * @param stride
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

typedef struct PointOfInterest {
    size_t x;
    size_t y;
    unsigned int quality;
} PointOfInterest;

typedef struct POINode {
    PointOfInterest poi;
    struct POINode *next;
    struct POINode *prev;
} POINode;

/**
 * Find features to feed to Lucas Kanade
 * 
 * @param image a single frame input
 * @param corners PointOfInterest coordinate array of corners detected
 * @param max_corners maximum number of corners to return
 * @param quality_level minimal acceptable quality level of corners, 
 * @param min_distance minimal euclidian distance between corners detected
 * @param mask optional region of interest
 * @param block_size size of an average block for computing a derivative covariation matrix over each pixel neighborhood
 */
void ShiTomasiCornerDetection(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_context context,
    cl_mem *image, 
    cl_mem *I_x,
    cl_mem *I_y,
    const int width,
    const int height,
    PointOfInterest *corners, 
    int max_corners, 
    double quality_level, 
    double min_distance, 
    bool *mask, 
    int block_size
) 
{
    size_t res_size = width * height * sizeof(float);

    // allocate memory for minimum eigenvalue of structure tensor
    float *st_response = (float *) malloc(res_size);
    memset(st_response, 0, res_size);

    cl_int err;
    cl_mem device_st_response = clCreateBuffer(context, CL_MEM_READ_WRITE, res_size, st_response, &err);
    CHECK_ERR(err, "clCreateBuffer");

    err = clEnqueueWriteBuffer(queue, device_st_response, CL_BLOCKING, 0, res_size, st_response, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer");

    // __kernel void shiTomasi(
    // __global const float *I_x, __global const float *I_y, __global float *response,
    // const int width, const int height, const int block_size)
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), I_x);
    CHECK_ERR(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), I_y);
    CHECK_ERR(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_st_response);
    CHECK_ERR(err, "clSetKernelArg 2");
    err = clSetKernelArg(kernel, 3, sizeof(int), &width);
    CHECK_ERR(err, "clSetKernelArg 3");
    err = clSetKernelArg(kernel, 4, sizeof(int), &height);
    CHECK_ERR(err, "clSetKernelArg 4");
    err = clSetKernelArg(kernel, 5, sizeof(int), &block_size);
    CHECK_ERR(err, "clSetKernelArg 5");

    size_t local_work_size[WORK_DIM] = {LWS, LWS};
    size_t global_work_size[WORK_DIM] = {
        ((width + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0],
        ((height + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1]
    };

    clEnqueueNDRangeKernel(queue, kernel, WORK_DIM, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, device_st_response, CL_BLOCKING, 0, res_size, st_response, 0, NULL, NULL);

    // find maximum value 
    // - max * threshold is minimum quality level
    float max = 0;
    for (int i = 0; i < width * height; ++i) {
        if (st_response[i] > max) {
            max = st_response[i];
        }
    }

    float threshold = max * quality_level;
    struct POINode *pois = (struct POINode *)malloc(sizeof(POINode));
    pois->next = NULL;
    pois->prev = NULL;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width * height; ++j) {
            if (st_response[i] > threshold) {
                // insert at head of linked list
                pois->prev = (struct POINode *)malloc(sizeof(struct POINode));
                pois->prev->next = pois;
                pois = pois->prev;
                pois->prev = NULL;
                pois->poi.x = i;
                pois->poi.y = j;
                pois->poi.quality = st_response[i * width + j];
            }
        }
    }

    // sort linked list

    // filter by euclidian distance
    for (POINode *itr = pois; itr != NULL; itr = itr->next) {
        for (POINode *cmpitr = pois; cmpitr != NULL; cmpitr = cmpitr->next) {
            // if itr is close to cmpitr and 
        }
    }

    clReleaseMemObject(device_st_response);
    free(st_response);
}

void Solver(
    cl_command_queue queue,
    cl_kernel stack_kernel,
    cl_kernel invert_kernel,
    cl_context context,
    cl_mem *image, 
    cl_mem *I_x,
    cl_mem *I_y,
    cl_mem *I_t,
    cl_mem *u,
    cl_mem *v,
    const int width,
    const int height,
    int K
)  
{
    // Construct A as [ I_x(q_1), I_y(q_1) \\ I_x(q_2) I_y(q_2) \\ ... ]
    // centered aroud p
    // v = [V_x \\ V_y] 
    // b = [-I_t(q_1) \\ - I_t(q_2) \\ ... ]
    cl_int err;
    
    int o_width = width - K + 1;
    int o_height = height - K + 1;

    int A_sz = 2 * K * K * o_width * o_height;
    int b_sz = 1 * K * K * o_width * o_height;

    cl_mem device_A = clCreateBuffer(context, CL_MEM_READ_WRITE, A_sz * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer");

    cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, b_sz * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer");

    // __kernel void constructLKVector(
    // __global const float * I_x, __global const float * I_y, __global const float * I_t,
    // __global float * A, __global float * b,
    // const int width, const int height, const int window)
    err = clSetKernelArg(stack_kernel, 0, sizeof(cl_mem), I_x);
    CHECK_ERR(err, "clSetKernelArg 0");
    err = clSetKernelArg(stack_kernel, 1, sizeof(cl_mem), I_y);
    CHECK_ERR(err, "clSetKernelArg 1");
    err = clSetKernelArg(stack_kernel, 2, sizeof(cl_mem), I_t);
    CHECK_ERR(err, "clSetKernelArg 2");
    err = clSetKernelArg(stack_kernel, 3, sizeof(cl_mem), &device_A);
    CHECK_ERR(err, "clSetKernelArg 3");
    err = clSetKernelArg(stack_kernel, 4, sizeof(cl_mem), &device_b);
    CHECK_ERR(err, "clSetKernelArg 4");
    err = clSetKernelArg(stack_kernel, 5, sizeof(int), &width);
    CHECK_ERR(err, "clSetKernelArg 5");
    err = clSetKernelArg(stack_kernel, 6, sizeof(int), &height);
    CHECK_ERR(err, "clSetKernelArg 6");
    err = clSetKernelArg(stack_kernel, 7, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg 7");

    // do an im2col type construction of our Gemm input
    const size_t B = o_width * o_height;
    size_t gws[3] = {((B + 15) / 16) * 16, K, K};
    size_t lws[3] = {16, 1, 1};

    err = clEnqueueNDRangeKernel(queue, stack_kernel, 3, 0, gws, lws, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");

    // solution is v = (A^T A)^(-1) A^T b

    const size_t m = 2;
    const size_t n = 2;
    const size_t k = K * K;

    std::vector<size_t> a_offsets = std::vector<size_t>(B, 0);
    std::vector<size_t> ata_offsets = std::vector<size_t>(B, 0);
    std::vector<size_t> ata_inv_at_offsets = std::vector<size_t>(B, 0);
    std::vector<size_t> b_offsets = std::vector<size_t>(B, 0);
    std::vector<size_t> v_offsets = std::vector<size_t>(B, 0);

    for (size_t i = 0; i < B; ++i) {
        a_offsets[i] = i * K * K * 2;
        ata_offsets[i] = i * 2 * 2;
        ata_inv_at_offsets[i] = i * K * K * 2;
        b_offsets[i] = i * K * K;
        v_offsets[i] = 2 * i;
    }

    std::vector<float> alphas = std::vector<float>(B, 1.0f);
    std::vector<float> betas = std::vector<float>(B, 0.0f);
    
    cl_mem device_ATA = clCreateBuffer(context, CL_MEM_READ_WRITE, B * 4 * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer");

    clblast::StatusCode cl_err = clblast::GemmBatched(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kYes,
        clblast::Transpose::kNo,
        m, n, k,
        alphas.data(),
        device_A, a_offsets.data(), 2,
        device_A, a_offsets.data(), 2,
        betas.data(), 
        device_ATA, ata_offsets.data(), 2,
        B, &queue
    );
    CHECK_ERR((cl_int)cl_err, "GemmBatched");
    clblast::ClearCache();

    // invert A^T A
    err = clSetKernelArg(invert_kernel, 0, sizeof(cl_mem), &device_ATA);
    CHECK_ERR(err, "clSetKernelArg");

    size_t gws_inv[1] = {B};
    size_t lws_inv[1] = {1};
    err = clEnqueueNDRangeKernel(queue, invert_kernel, 1, 0, gws_inv, lws_inv, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");

    // Calculate (A^T A)^(-1) A^T
    cl_mem device_ATA_inv_AT = clCreateBuffer(context, CL_MEM_READ_WRITE, B * 2 * K * K * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer");

    cl_err = clblast::GemmBatched(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo,
        clblast::Transpose::kYes,
        2, K * K, 2,
        alphas.data(),
        device_ATA, ata_offsets.data(), 2,
        device_A, a_offsets.data(), 2,
        betas.data(),
        device_ATA_inv_AT, ata_inv_at_offsets.data(), K * K,
        B, &queue
    );
    CHECK_ERR((cl_int)cl_err, "GemmBatched");
    clblast::ClearCache();

    // Calculate (A^T A)^(-1) A^T
    cl_mem device_v = clCreateBuffer(context, CL_MEM_READ_WRITE, B * 2 * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer");

    cl_err = clblast::GemmBatched(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo,
        clblast::Transpose::kNo,
        2, 1, K * K,
        alphas.data(),
        device_ATA_inv_AT, ata_inv_at_offsets.data(), K*K,
        device_b, b_offsets.data(), 1,
        betas.data(),
        device_v, v_offsets.data(), 1,
        B, &queue
    );

    clblast::ClearCache();
    clReleaseMemObject(device_A);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_ATA);
    clReleaseMemObject(device_ATA_inv_AT);
}

void OpenCLOpticalFlow(Matrix *input0, Matrix *input1, Matrix *result_x, Matrix *result_y)
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
    cl_kernel stack_kernel;          // kernel
    cl_kernel invert_kernel;

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    //@@ define local and global work sizes
    size_t local_item_size[2] = {16, 16};

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
    stack_kernel = clCreateKernel(program, "constructLKVector", &err);
    CHECK_ERR(err, "clCreateKernel");
    invert_kernel = clCreateKernel(program, "inPlaceInvert2x2Matrix", &err);
    CHECK_ERR(err, "clCreateKernel");

    printf("Kernels created\n");
    
    // Allocate GPU here
    int height = input0->shape[0];
    int width = input0->shape[1];

    // Output
    cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer");

    // stores frames on a pyramid layer basis
    cl_mem frame1s[PYR_LAYERS];
    cl_mem frame2s[PYR_LAYERS];
    cl_mem frame1_Ix[PYR_LAYERS];
    cl_mem frame1_Iy[PYR_LAYERS];
    cl_mem frame2_Ix[PYR_LAYERS];
    cl_mem frame2_Iy[PYR_LAYERS];
    cl_mem It[PYR_LAYERS];
    cl_mem u[PYR_LAYERS];
    cl_mem v[PYR_LAYERS];

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
        u[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
        CHECK_ERR(err, "clCreateBuffer");
        v[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width / pow(pow(DSR, i), 2) * sizeof(float), NULL, &err);
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
        // Gaussian applied with downsampling rate stride to prevent aliasing
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

    // determine points to track
    for (int i = 0; i < PYR_LAYERS; ++i) {
        continue;
    }

    // calculate temporal gradient
    for (int i = 0; i < PYR_LAYERS; i++) {
        CalculateTemporalGradient(
            queue, temporal_gradient_kernel, WORK_DIM, local_item_size, &frame1s[i], &frame2s[i], &It[i], width / pow(DSR, i), height / pow(DSR, i)
        );
    }

    for (int i = 0; i < PYR_LAYERS; i++) {
        Solver(
            queue, stack_kernel, invert_kernel, context, &frame1s[i], &frame1_Ix[i], &frame1_Iy[i], &It[i], &u[i], &v[i], width, height, 7 
        );
    }

    //@@ Copy the GPU memory back to the CPU here
    /*err = clEnqueueReadBuffer(queue, It[PYR_LAYERS - 1], CL_TRUE, 0, height * width / pow(pow(DSR, PYR_LAYERS - 1), 2) * sizeof(float), result->data, 0, NULL, NULL);
    result->shape[0] = height / pow(DSR, PYR_LAYERS - 1);
    result->shape[1] = width / pow(DSR, PYR_LAYERS - 1);*/

    err = clEnqueueReadBuffer(queue, u[0], CL_TRUE, 0, height * width * sizeof(float), result_x->data, 0, NULL, NULL);
    result_x->shape[0] = height;
    result_y->shape[1] = width;
    CHECK_ERR(err, "clEnqueueReadBuffer");

    err = clEnqueueReadBuffer(queue, v[0], CL_TRUE, 0, height * width * sizeof(float), result_x->data, 0, NULL, NULL);
    result_x->shape[0] = height;
    result_y->shape[1] = width;
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
        clReleaseMemObject(u[i]);
        clReleaseMemObject(v[i]);
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
    Matrix host_a, host_b, host_c, host_d;
    
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

    host_d.shape[0] = output_height;
    host_d.shape[1] = output_width;
    host_d.data = (float *)malloc(sizeof(float) * host_d.shape[0] * host_d.shape[1]);

    // Call your optical flow.
    printf("Start optical flow...\n");
    OpenCLOpticalFlow(&host_a, &host_b, &host_c, &host_d);

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
            img.Pixel(x, y, 1) = host_d.data[y * input_width + x];
        }
    }
    WriteFlowFile(img, "output.flo");

    std::system("../../helper_lib/flow-code/color_flow output.flo outputcolors.png");

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    //free(answer.data);
    printf("Done.\n");
    return 0;
}
