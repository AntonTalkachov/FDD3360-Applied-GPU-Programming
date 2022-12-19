#include <stdio.h>
#include <sys/time.h>
#define TPB 256
#define ARRAY_SIZE 10000

__host__ double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void distanceKernel(float a, float *d_x, float *d_y, float *d_z_GPU)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= ARRAY_SIZE) return;
  d_z_GPU[i] = a * d_x[i] + d_y[i];
}

int main()
{
    float *x = (float*)malloc(ARRAY_SIZE*sizeof(float));
    float *y = (float*)malloc(ARRAY_SIZE*sizeof(float));
    float *z_CPU = (float*)malloc(ARRAY_SIZE*sizeof(float));
    float *z_GPU = (float*)malloc(ARRAY_SIZE*sizeof(float));
    
    float a;
    
    a = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    //printf("a = %f\n",a);

    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        y[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        //printf("x[%d] = %f, y[%d] = %f \n", i , x[i], i , y[i]);
    }

    double iStart = cpuSecond();
    
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        z_CPU[i] = a * x[i] + y[i];
    }

    double iElaps = cpuSecond() - iStart;

    printf("Computing SAXPY on the CPU… Done!\nIt took %f ms\n", iElaps*1000);

    
    float *d_x, *d_y, *d_z_GPU;
    cudaMalloc(&d_x, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&d_y, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&d_z_GPU, ARRAY_SIZE*sizeof(float));
    
    cudaMemcpy(d_x, x, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    iStart = cpuSecond();
    distanceKernel<<<(ARRAY_SIZE + TPB - 1)/TPB, TPB>>>(a, d_x, d_y, d_z_GPU);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Computing SAXPY on the GPU… Done!\nIt took %f ms\n", iElaps*1000);

    cudaMemcpy(z_GPU, d_z_GPU, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z_GPU);

    int j = 0;
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        //printf("GPU = %f, CPU = %f, diff = %f \n",z_GPU[i],z_CPU[i], z_GPU[i]-z_CPU[i]);
        if (abs(z_GPU[i]-z_CPU[i]) > 1e-6) j++;
    }
    if (j == 0) {printf("Comparing the output for each implementation… Correct!\n");}
    else {printf("Comparing the output for each implementation… Troubles :(\n");}

}