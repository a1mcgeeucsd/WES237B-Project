__kernel void lk_downsample(
    __global const float* src, 
    __global float* dst, 
    const int src_width, 
    const int src_height,
    const int dst_width, 
    const int dst_height) 
{
    int dst_x = get_global_id(0);
    int dst_y = get_global_id(1);
    
    // Bounds guard check
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    // Target coordinate center in source image space (doubled)
    int src_center_x = dst_x * 2;
    int src_center_y = dst_y * 2;
    
    // Separable 5x5 Gaussian weights components (Normalized sum = 1.0)
    const float w[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    
    float blurred_value = 0.0f;
    
    // Apply 5x5 local window sample loop
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            
            int sx = clamp(src_center_x + kx, 0, src_width - 1);
            int sy = clamp(src_center_y + ky, 0, src_height - 1);
            
            float pixel = src[sy * src_width + sx];
            
            blurred_value += pixel * w[kx + 2] * w[ky + 2];
        }
    }
    
    // Write out results to the smaller pyramid destination buffer
    dst[dst_y * dst_width + dst_x] = blurred_value;
}
