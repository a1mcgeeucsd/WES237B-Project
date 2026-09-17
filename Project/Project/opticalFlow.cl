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

    // Bounds guard check 
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