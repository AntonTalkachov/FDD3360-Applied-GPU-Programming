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

__global__ void distanceKernel(int NUM_PARTICLES, Particle *particles_GPU, float dt, int NUM_ITERATIONS)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;

    for (int k = 0; k < NUM_ITERATIONS; k++)
    {
        particles_GPU[i].velocity.x = 3.81* particles_GPU[i].velocity.x *(1 - particles_GPU[i].velocity.x);  //  updating velocities
        particles_GPU[i].velocity.y = 3.81* particles_GPU[i].velocity.y *(1 - particles_GPU[i].velocity.y);
        particles_GPU[i].velocity.z = 3.81* particles_GPU[i].velocity.z *(1 - particles_GPU[i].velocity.z);

        particles_GPU[i].position.x = particles_GPU[i].position.x + dt * particles_GPU[i].velocity.x;
        particles_GPU[i].position.y = particles_GPU[i].position.y + dt * particles_GPU[i].velocity.y;
        particles_GPU[i].position.z = particles_GPU[i].position.z + dt * particles_GPU[i].velocity.z;
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) return;
    const int NUM_PARTICLES = strtol(argv[1], nullptr, 0);
    const int NUM_ITERATIONS = strtol(argv[2], nullptr, 0);
    const int BLOCK_SIZE = strtol(argv[3], nullptr, 0);
    int j = 0;
    
    float dt = 1.;
    Particle *particles = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
    Particle *particles_check = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
    Particle *particles_GPU;    
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        particles[i].position.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        particles[i].position.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        particles[i].position.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        particles[i].velocity.x = abs(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        particles[i].velocity.y = abs(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        particles[i].velocity.z = abs(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));

        /*printf("Particle %d position is {%f, %f, %f}.  ",i, particles[i].position.x,
        particles[i].position.y, particles[i].position.z);
        printf("It's velocity is {%f, %f, %f}\n", particles[i].velocity.x,
        particles[i].velocity.y, particles[i].velocity.z);*/
    }
    
    double iStart = cpuSecond();
    cudaMalloc(&particles_GPU, NUM_PARTICLES*sizeof(Particle));
    cudaMemcpy(particles_GPU, particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);    
    for (int k = 0; k < NUM_ITERATIONS; k++)
    {    
        distanceKernel<<<(NUM_PARTICLES + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(NUM_PARTICLES, particles_GPU, dt, 1);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(particles_check, particles_GPU, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(particles_GPU);

    double iElapsGPU = cpuSecond() - iStart;
    printf("Computing SAXPY on the GPU (all interations inside GPU)… Done!\nIt took %f ms\n", iElapsGPU*1000);

    iStart = cpuSecond();
    cudaMalloc(&particles_GPU, NUM_PARTICLES*sizeof(Particle));
    cudaMemcpy(particles_GPU, particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);     
    distanceKernel<<<(NUM_PARTICLES + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(NUM_PARTICLES, particles_GPU, dt, NUM_ITERATIONS);
    cudaDeviceSynchronize();
    cudaMemcpy(particles_check, particles_GPU, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(particles_GPU);
    
    iElapsGPU = cpuSecond() - iStart;
    printf("Computing SAXPY on the GPU (all interations inside GPU)… Done!\nIt took %f ms\n", iElapsGPU*1000);


    iStart = cpuSecond();   
    for (int k = 0; k < NUM_ITERATIONS; k++)
        for (int i = 0; i < NUM_PARTICLES; i++)
        {
            particles[i].velocity.x = 3.81 * particles[i].velocity.x * (1 - particles[i].velocity.x);  //  updating velocities
            particles[i].velocity.y = 3.81 * particles[i].velocity.y * (1 - particles[i].velocity.y);
            particles[i].velocity.z = 3.81 * particles[i].velocity.z * (1 - particles[i].velocity.z);

            particles[i].position.x = particles[i].position.x + dt * particles[i].velocity.x;
            particles[i].position.y = particles[i].position.y + dt * particles[i].velocity.y;
            particles[i].position.z = particles[i].position.z + dt * particles[i].velocity.z;
        }

    double iElapsCPU = cpuSecond() - iStart;
    printf("Computing SAXPY on the CPU… Done!\nIt took %f ms\n", iElapsCPU*1000);

    for (int i = 0; i < NUM_PARTICLES; i++)
        if (abs(particles[i].position.x-particles_check[i].position.x) > 1e-5 ||
            abs(particles[i].position.y-particles_check[i].position.y) > 1e-5 ||
            abs(particles[i].position.z-particles_check[i].position.z) > 1e-5
            ) j++;

    if (j == 0)
        printf("Comparing the output for each implementation… Correct!\n");
    else
        printf("Comparing the output for each implementation… Troubles :(\n");

    return 0;
}