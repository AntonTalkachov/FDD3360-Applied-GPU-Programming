#include <stdio.h>
#define TPB 256


__global__ void distanceKernel()
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  printf("Hello World! My threadId is %d.\n", i);
}

int main()
{
  distanceKernel<<<1, TPB>>>();
  cudaDeviceSynchronize();

  return 0;
}
