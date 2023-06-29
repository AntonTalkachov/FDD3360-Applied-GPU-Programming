#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include <curand.h>
#define BLOCK_SIZE 128

typedef struct
{
   float3 position, velocity;
} Particle;

__host__ double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void distanceKernel(int NUM_PARTICLES, Particle *particles_GPU, float dt, int offset)
{
    const int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;
        particles_GPU[i].velocity.x = 3.81* particles_GPU[i].velocity.x *(1 - particles_GPU[i].velocity.x);  //  updating velocities
        particles_GPU[i].velocity.y = 3.81* particles_GPU[i].velocity.y *(1 - particles_GPU[i].velocity.y);
        particles_GPU[i].velocity.z = 3.81* particles_GPU[i].velocity.z *(1 - particles_GPU[i].velocity.z);

        particles_GPU[i].position.x = particles_GPU[i].position.x + dt * particles_GPU[i].velocity.x;
        particles_GPU[i].position.y = particles_GPU[i].position.y + dt * particles_GPU[i].velocity.y;
        particles_GPU[i].position.z = particles_GPU[i].position.z + dt * particles_GPU[i].velocity.z;
}

int main(int argc, char* argv[])
{
    if (argc < 3) return;
    const int NUM_PARTICLES = strtol(argv[1], nullptr, 0);
    const int NUM_ITERATIONS = strtol(argv[2], nullptr, 0);
    const int nStreams = strtol(argv[3], nullptr, 0);
    
    const int streamSize = NUM_PARTICLES / nStreams;
    float dt = 1.;
    Particle *particles, *particles_GPU;
    cudaMallocHost((void **) &particles, NUM_PARTICLES*sizeof(Particle), cudaHostAllocDefault);

    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        particles[i].position.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        particles[i].position.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        particles[i].position.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        particles[i].velocity.x = abs(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        particles[i].velocity.y = abs(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        particles[i].velocity.z = abs(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
    }
    
    double iStart = cpuSecond();
    
    cudaMalloc((void**)&particles_GPU, NUM_PARTICLES*sizeof(Particle));

    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreate(&stream[i]);
 
    for (int k = 0; k < NUM_ITERATIONS; k++)
        for (int i = 0; i < nStreams; i++)
        {
            int offset = i * streamSize;
            cudaMemcpyAsync(&particles_GPU[offset], &particles[offset], streamSize*sizeof(Particle), cudaMemcpyHostToDevice, stream[i]);
            distanceKernel<<<(streamSize + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream[i]>>>(NUM_PARTICLES, particles_GPU, dt, offset);
            cudaMemcpyAsync(&particles[offset], &particles_GPU[offset], streamSize*sizeof(Particle), cudaMemcpyDeviceToHost, stream[i]);
        }

    for (int i = 0; i < nStreams; i++)
        cudaStreamDestroy(stream[i]);
    cudaFree(particles_GPU);
    cudaFreeHost(particles);

    double iElapsGPU = cpuSecond() - iStart;
    printf("Computing on the GPU with %d streams… Done!\nIt took %f ms\n", nStreams, iElapsGPU*1000);
    


    //Exercise with batches and one stream
    if (argc == 5)
    {
    const int num_batches = strtol(argv[4], nullptr, 0);
    int batch_size = NUM_PARTICLES / num_batches;
    cudaMallocHost((void **) &particles, NUM_PARTICLES*sizeof(Particle), cudaHostAllocDefault);
    
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        particles[i].position.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        particles[i].position.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        particles[i].position.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        particles[i].velocity.x = abs(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        particles[i].velocity.y = abs(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        particles[i].velocity.z = abs(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
    }
    

    iStart = cpuSecond();
    cudaMalloc((void**)&particles_GPU, NUM_PARTICLES*sizeof(Particle));
    
    for (int k = 0; k < NUM_ITERATIONS; k++)
        for (int i = 0; i < num_batches; i++)
        {
            int offset = i * batch_size;
            cudaMemcpy(&particles_GPU[offset], &particles[offset], batch_size*sizeof(Particle), cudaMemcpyHostToDevice);
            distanceKernel<<<(batch_size + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(NUM_PARTICLES, particles_GPU, dt, offset);
            cudaMemcpy(&particles[offset], &particles_GPU[offset], batch_size*sizeof(Particle), cudaMemcpyDeviceToHost);
        }
    
    cudaFree(particles_GPU);
    cudaFreeHost(particles);

    iElapsGPU = cpuSecond() - iStart;
    printf("Computing on the GPU… Done!\nIt took %f ms\n", iElapsGPU*1000);

    return 0;
    }
}
