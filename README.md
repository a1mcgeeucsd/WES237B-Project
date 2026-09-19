# Lucas-Kanade Optical Flow
## WES 237B Project
Joe Fields & Alex McGee

### Kernels used:

`conv2d`: Used for downsampling with a Gaussian filter, and XY Gradients with Sobel filters.

`temporalGradient`: Element-wise subtraction to get the I_t (temporal gradient) frame

`constructLKVector`: Constructs A & b vectors for Av = b least-squares with Gemm.

`inPlaceInver2x2Matrix`: Calculates the inverse of (A^TA) matrix

`upsample_kernel`: Upsamples motion vector from one level to the other. Scales values as well to account for larger pixel shift.

`unterleave`: "Un-interleaves" our interleaved XY values into an X and Y matrix (u, v). Accumulates values so that we can refine our estimate.


### Main.cpp functions:

`Convolve()`: Loads and launches convolution kernel with frame and filter. This is used for Gaussian downsample as well as X and Y gradients.

`CalculateTemporalGradient()`: Loads and launches temporal gradient kernel.

`Upsample()`: Loads and launches upsample kernel. Converts u,v[coarse] estimate into u,v[fine] estimate.

`Warp()`: Loads and launches warp kernel. Uses u,v[] fields to warp Frame 2 to estimate Frame 1. We can then iteratively solve to refine our estimates.

`Solver()`: Runs `constructLKVector` a.k.a. `stack_kernel` to calculate A, b. Multiplies A by A_transpose. Runs `inPlaceInvert2x2Matrix` to calculate (ATA)^(-1). Multiplies by b vector to calculate u, v, runs `unterleave` kernel to un-interleave u, v vectors.

`OpenCLOpticalFlow()`: Recurses through framework: `WARPS` warps in each of `PYR_LAYERS` layers to nail down optical flow estimate.

`main()`: Writes to .flo, calls Middlebury `color_flow` to convert .flo file into a color image `output.png`. 

### Others:

/tools/ has AI-generated python scripts that I used to benchmark our .flo file.

/helper_lib/flow-code/ has the Middlebury C++ functions to validate and transform .flo files. `color_flow` is called via system call in our main function.
