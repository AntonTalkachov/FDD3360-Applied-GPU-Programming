#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void SAXPY_CPU(int N, float *x, float *y, float a, float *res)
{
    for (int i = 0; i < N; i++)
        res[i] = a*x[i] + y[i];
}

void SAXPY_GPU(int N, float *x, float *y, float a)
{
    #pragma acc data copyin(x[0:N]) copyin(y[0:N]) copyout(y[0:N])
    {
        #pragma acc parallel loop
            for (int i = 0; i < N; i++)
                y[i] = a*x[i] + y[i];
    }
}

int main(int argc, char *argv[]) {
  if (argc != 2)
  {
    printf("Input parameter array size\n");
    return 1;
  }
  int ARRAY_SIZE = atoi(argv[1]);

  float x[ARRAY_SIZE], y[ARRAY_SIZE], CPU_res[ARRAY_SIZE];
  float a = (float)rand()/(float)(RAND_MAX);
  for (int i = 0; i < ARRAY_SIZE; i++) 
  {
      x[i] = (float)rand()/(float)(RAND_MAX);
      y[i] = (float)rand()/(float)(RAND_MAX);
  }
  
  double iStart = cpuSecond();
  SAXPY_CPU(ARRAY_SIZE, (void *) &x,(void *) &y,a,(void *) &CPU_res);
  double iElaps = cpuSecond() - iStart;
  printf("Computing SAXPY on the CPU… Done!\nIt took %f ms\n", iElaps*1000);
  
  iStart = cpuSecond();
  SAXPY_GPU(ARRAY_SIZE, (void *) &x,(void *) &y,a);
  iElaps = cpuSecond() - iStart;
  printf("Computing SAXPY on the GPU… Done!\nIt took %f ms\n", iElaps*1000);

  /* Check that result is correct */
  printf("Comparing the output for each implementation… ");
  int k = 0;
  for (int i = 0; i < ARRAY_SIZE; i++) 
    if (abs(y[i]/ CPU_res[i] - 1.) > 1.e-10)
    {
      printf("Error at %d (%f /= %f)\n", i, y[i], CPU_res[i]);
      k++;
    }
  if (k == 0) printf("Correct!\n");
  
  return 0;
}
