#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>


__host__ double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void distanceKernel(int NUM_ITER, int *array, curandState *states)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= NUM_ITER) return;

  double x,y,z;

  int seed = i; // different seed per thread
  curand_init(seed, i, 0, &states[i]);  // 	Initialize CURAND

  x = curand_uniform (&states[i]);
  y = curand_uniform (&states[i]);
  z = sqrt((x*x) + (y*y));
        
  // Check if point is in unit circle
  if (z <= 1.0)
    array[i] = 1;
  else
    array[i] = 0;
}

int main(int argc, char* argv[])
{
    for(int BLOCK_SIZE = 32; BLOCK_SIZE < 33; BLOCK_SIZE*=2)
    for(int NUM_ITER = 1000; NUM_ITER < 2e8; NUM_ITER*=2)
    {
    
    double iStart = cpuSecond();
    int count = 0;
    int *counts_array = (int*)malloc(NUM_ITER*sizeof(int));
    int *d_counts_array;

    curandState *dev_random;
    cudaMalloc((void**)&dev_random, NUM_ITER*sizeof(curandState));

    cudaMalloc(&d_counts_array, NUM_ITER*sizeof(int)); 

    distanceKernel<<<(NUM_ITER + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(NUM_ITER, d_counts_array, dev_random);
    cudaDeviceSynchronize();
    cudaMemcpy(counts_array, d_counts_array, NUM_ITER*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < NUM_ITER; i++)
      if(counts_array[i] == 1) count++;

    // Estimate Pi and display the result
    double pi = ((double)count / (double)NUM_ITER) * 4.0;

    double iElapsGPU = cpuSecond() - iStart;
    
    //printf("The result is %f\n", pi);
    printf("{%d,%.10f,%f,%d},",NUM_ITER, pi, iElapsGPU*1000, BLOCK_SIZE);

    cudaFree(d_counts_array);
    cudaFree(dev_random);
    }
    
    return 0;
}