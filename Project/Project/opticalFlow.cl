__kernel void opticalFlow(
    __global const unsigned int *A, __global unsigned int *C,
    const unsigned int numARows, const unsigned int numAColumns) {
  //@@ Compute C = AB 
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < numAColumns && y < numARows) {
    C[y * numAColumns + x] = A[y * numAColumns + x];
  }
}
