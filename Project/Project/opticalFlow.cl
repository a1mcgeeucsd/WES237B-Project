__kernel void conv2d(
    __global const float* src, 
    __global float* dst, 
    __global const float* filter_weights,
    const int kernel_radius,
    const int stride,
    const int src_width, 
    const int src_height, 
    const int dst_width, 
    const int dst_height) 
{ 
    int dst_x = get_global_id(0); 
    int dst_y = get_global_id(1); 

    if (dst_x >= dst_width || dst_y >= dst_height) return; 

    // Calculate center point in source image based on downsampling stride
    int src_center_x = dst_x * stride; 
    int src_center_y = dst_y * stride; 

    int kernel_width = (2 * kernel_radius) + 1;
    float accumulated_value = 0.0f; 

    // Apply dynamic local window sample loop 
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) { 
        int sy = clamp(src_center_y + ky, 0, src_height - 1);
        int src_row_offset = sy * src_width;
        
        int filter_row_offset = (ky + kernel_radius) * kernel_width;

        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) { 
            int sx = clamp(src_center_x + kx, 0, src_width - 1); 
            
            float pixel = src[src_row_offset + sx]; 
            
            // Map 2D loop indices to 1D filter array index
            float weight = filter_weights[filter_row_offset + (kx + kernel_radius)]; 
            
            accumulated_value += pixel * weight; 
        } 
    } 

    // Write out results to destination buffer 
    dst[dst_y * dst_width + dst_x] = accumulated_value; 
}

__kernel void temporalGradient(
    __global const float* src1, 
    __global const float* src2, 
    __global float* dst,
    const int src_width, 
    const int src_height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < src_width && y < src_height) {
        dst[y * src_width + x] = src2[y * src_width + x] - src1[y * src_width + x];
    }
}

__kernel void hadamardProduct(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int width,
    const int height
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        C[y * width + x] = A[y * width + x] * B[y * width + x];
    }
}

__kernel void shiTomasi(
    __global const float *I_x,
    __global const float *I_y,
    __global float *response,
    const int width,
    const int height,
    const int block_size
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int radius = block_size / 2;

    if (x >= width || y >= height) {
        return;
    }

    // construct [A B \\ B C]
    float A = 0;
    float B = 0;
    float C = 0;
    float l1, l2;

    // calculate I_x^2, I_x * I_y, I_y^2 over block neighborhood
    for (int i = -radius; i < radius; ++i) {
        if ((x + i) < 0 || (x + i) >= height) {
            continue;
        }

        for (int j = -radius; j < radius; ++j) {
            if ((y + j) < 0 || (y + j) >= width) {
                continue;
            }
            A += I_x[(y + i) * width + x + j] * I_x[(y + i) * width + x + j];
            B += I_x[(y + i) * width + x + j] * I_y[(y + i) * width + x + j];
            C += I_y[(y + i) * width + x + j] * I_y[(y + i) * width + x + j];
        }
    }

    // find eigenvalues
    l1 = (A + C + sqrt((A - C) * (A - C) + 4 * B * B)) / 2;
    l2 = (A + C - sqrt((A - C) * (A - C) + 4 * B * B)) / 2;

    response[y * width + x] = (l1 < l2) ? l1 : l2;
}

__kernel void constructLKVector(
    __global const float *I_x,
    __global const float *I_y,
    __global const float *I_t,
    __global float *A,
    __global float *b,
    const int width,
    const int height,
    const int window
)
{
    int batch = get_global_id(0);
    int k1   = get_global_id(1);
    int k2   = get_global_id(2);

    int radius = window / 2;

    int o_width  = width  - window + 1;
    int o_height = height - window + 1;

    // Position of the center pixel
    int out_x = batch % o_width;
    int out_y = batch / o_width;

    int center_x = out_x + radius;
    int center_y = out_y + radius;

    // Position inside the window
    int x = center_x + k2 - radius;
    int y = center_y + k1 - radius;

    // Shouldn't be necessary for a valid window,
    // but harmless as a safety check.
    if (x < 0 || x >= width ||
        y < 0 || y >= height)
    {
        return;
    }

    int image_idx = y * width + x;

    int window_idx = k1 * window + k2;

    // A[batch] is a window^2 x 2 matrix
    int A_idx = (batch * window * window + window_idx) * 2;

    A[A_idx + 0] = I_x[image_idx];
    A[A_idx + 1] = I_y[image_idx];

    // b[batch] is a window^2 x 1 vector
    int b_idx = batch * window * window + window_idx;

    b[b_idx] = -I_t[image_idx];
}

__kernel void inPlaceInvert2x2Matrix(
    __global float * ATA
)
{
    int batch_id = get_global_id(0);

    float a = ATA[batch_id * 4 + 0];
    float b = ATA[batch_id * 4 + 1];
    float c = ATA[batch_id * 4 + 2];
    float d = ATA[batch_id * 4 + 3];

    float det = a * d - b * c;
    if (det < 1e-6f && det > -1e-6f) {
        ATA[batch_id * 4 + 0] = ATA[batch_id * 4 + 1] = ATA[batch_id * 4 + 2] = ATA[batch_id * 4 + 3] = 0.0f;
        return;
    }

    ATA[batch_id * 4 + 0] = d / det;
    ATA[batch_id * 4 + 1] = -b / det;
    ATA[batch_id * 4 + 2] = -c / det;
    ATA[batch_id * 4 + 3] = a / det;
}

    __kernel void backwardWarpBilinear(
        __global const float *I2,   // Second input frame to be warped
        __global const float *u,    // Current horizontal flow field (velocity)
        __global const float *v,    // Current vertical flow field (velocity)
        __global float *I2_warped,  // Output: Frame 2 aligned to Frame 1's coordinate space
        const int width,
        const int height
    ) {
        int x = get_global_id(0);
        int y = get_global_id(1);

        // Boundary check
        if (x >= width || y >= height) {
            return;
        }

        int idx = y * width + x;

        // Calculate source floating-point coordinate in Frame 1
        float src_x = (float)x + u[idx];
        float src_y = (float)y + v[idx];

        // Out-of-bounds check: if the flow points outside the image, fill with zero (or boundary pixel)
        if (src_x < 0.0f || src_x > (float)(width - 1) || 
            src_y < 0.0f || src_y > (float)(height - 1)) {
            I2_warped[idx] = 0.0f; 
            return;
        }

        // Find the 4 neighboring integer coordinates for bilinear interpolation
        int x0 = (int)floor(src_x);
        int y0 = (int)floor(src_y);
        int x1 = min(x0 + 1, width - 1);
        int y1 = min(y0 + 1, height - 1);

        // Calculate interpolation weights (distances to top-left neighbor)
        float tx = src_x - (float)x0;
        float ty = src_y - (float)y0;

        // Fetch the 4 neighbor pixel values from Frame 2
        float p00 = I2[y0 * width + x0]; // Top-Left
        float p10 = I2[y0 * width + x1]; // Top-Right
        float p01 = I2[y1 * width + x0]; // Bottom-Left
        float p11 = I2[y1 * width + x1]; // Bottom-Right

        // Perform bilinear interpolation
        float top    = p00 + tx * (p10 - p00);
        float bottom = p01 + tx * (p11 - p01);
        float final_pixel = top + ty * (bottom - top);

        // Write out the warped frame pixel
        I2_warped[idx] = final_pixel;
    }

    __kernel void upsample_kernel(
        __global const float *coarse_flow, // Input: u or v field from the lower-res level
        __global float *fine_flow,         // Output: Allocated buffer at the higher-res level
        const int coarse_width,
        const int coarse_height,
        const int fine_width,
        const int fine_height
    ) {
        // Each thread processes one unique pixel in the FINE (high-res) destination grid
        int fine_x = get_global_id(0);
        int fine_y = get_global_id(1);

        // Boundary check
        if (fine_x >= fine_width || fine_y >= fine_height) {
            return;
        }

        int fine_idx = fine_y * fine_width + fine_x;

        // Map fine grid coordinates back to coarse grid floating-point space.
        // Subtracting 0.25f and aligning scales properly centers the pixel positions 
        // across the 2x scale boundary, matching standard OpenCV/MATLAB pyramid rules.
        float coarse_src_x = (float)fine_x * 0.5f;
        float coarse_src_y = (float)fine_y * 0.5f;

        // Clamp coordinates to valid coarse image boundaries
        coarse_src_x = max(0.0f, min(coarse_src_x, (float)(coarse_width - 1)));
        coarse_src_y = max(0.0f, min(coarse_src_y, (float)(coarse_height - 1)));

        // Find the 4 neighboring integer pixels in the coarse grid
        int x0 = (int)floor(coarse_src_x);
        int y0 = (int)floor(coarse_src_y);
        int x1 = min(x0 + 1, coarse_width - 1);
        int y1 = min(y0 + 1, coarse_height - 1);

        // Calculate bilinear interpolation weights
        float tx = coarse_src_x - (float)x0;
        float ty = coarse_src_y - (float)y0;

        // Fetch the 4 coarse velocity samples
        float p00 = coarse_flow[y0 * coarse_width + x0]; // Top-Left
        float p10 = coarse_flow[y0 * coarse_width + x1]; // Top-Right
        float p01 = coarse_flow[y1 * coarse_width + x0]; // Bottom-Left
        float p11 = coarse_flow[y1 * coarse_width + x1]; // Bottom-Right

        // Bilinear interpolation
        float top    = p00 + tx * (p10 - p00);
        float bottom = p01 + tx * (p11 - p01);
        float coarse_velocity = top + ty * (bottom - top);

        // CRITICAL STEPS: Multiply by 2.0f because the image space has doubled.
        fine_flow[fine_idx] = coarse_velocity * 2.0f;
    }

__kernel void unterleave(
    __global const float *UV,
    __global float *u,
    __global float *v,
    const int width,
    const int height,
    const int K
)
{
    int radius = K / 2;

    int o_width  = width  - K + 1;
    int o_height = height - K + 1;

    int out_x = get_global_id(0);
    int out_y = get_global_id(1);

    // Only process valid convolution positions
    if (out_x >= o_width || out_y >= o_height)
        return;

    // Map valid output position to the center pixel
    int x = out_x + radius;
    int y = out_y + radius;

    // Row-major index in the original image
    int image_idx = y * width + x;

    // Row-major index in the valid convolution output
    int out_idx = out_y * o_width + out_x;

    // UV is interleaved: [u0, v0, u1, v1, ...]
    u[image_idx] += UV[2 * out_idx + 0];
    v[image_idx] += UV[2 * out_idx + 1];
}