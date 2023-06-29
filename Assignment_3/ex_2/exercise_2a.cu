#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include <curand.h>

typedef struct
{
   float3 position, velocity;
} Particle;

__host__ double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void distanceKernel(int NUM_PARTICLES, Particle *particles_GPU, float dt)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
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
    if (argc < 4) return;
    const int NUM_PARTICLES = strtol(argv[1], nullptr, 0);
    const int NUM_ITERATIONS = strtol(argv[2], nullptr, 0);
    const int BLOCK_SIZE = strtol(argv[3], nullptr, 0);
    const int SIMULATION_TYPE = strtol(argv[4], nullptr, 0);
    
    float dt = 1.;
    Particle *particles;
    if (SIMULATION_TYPE)
        particles = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
    else
    {   
        cudaMallocHost((void **) &particles, NUM_PARTICLES*sizeof(Particle), cudaHostAllocDefault);
    }
    Particle *particles_GPU; 
    
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
    cudaMalloc(&particles_GPU, NUM_PARTICLES*sizeof(Particle));
 
    for (int k = 0; k < NUM_ITERATIONS; k++)
    {    
        cudaMemcpy(particles_GPU, particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);   
        distanceKernel<<<(NUM_PARTICLES + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(NUM_PARTICLES, particles_GPU, dt);
        cudaDeviceSynchronize();
        cudaMemcpy(particles, particles_GPU, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);
    }
    cudaFree(particles_GPU);

    double iElapsGPU = cpuSecond() - iStart;
    if (SIMULATION_TYPE)
        printf("Computing SAXPY on the GPU cudaMalloc… Done!\nIt took %f ms\n", iElapsGPU*1000);
    else
        printf("Computing SAXPY on the GPU cudaMallocHost… Done!\nIt took %f ms\n", iElapsGPU*1000);

    return 0;
}