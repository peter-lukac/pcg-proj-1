/**
 * @File main.cu
 *
 * The main file of the project
 *
 * Paralelní programování na GPU (PCG 2020)
 * Projekt c. 1 (cuda)
 * Login: xlukac11
 */

#include <sys/time.h>
#include <cstdio>
#include <cmath>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  // Time measurement
  struct timeval t1, t2;

  if (argc != 10)
  {
    printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
    exit(1);
  }

  // Number of particles
  const int N           = std::stoi(argv[1]);
  // Length of time step
  const float dt        = std::stof(argv[2]);
  // Number of steps
  const int steps       = std::stoi(argv[3]);
  // Number of thread blocks
  const int thr_blc     = std::stoi(argv[4]);
  // Write frequency
  int writeFreq         = std::stoi(argv[5]);
  // number of reduction threads
  const int red_thr     = std::stoi(argv[6]);
  // Number of reduction threads/blocks
  const int red_thr_blc = std::stoi(argv[7]);

  // Size of the simulation CUDA gird - number of blocks
  const size_t simulationGrid = (N + thr_blc - 1) / thr_blc;
  // Size of the reduction CUDA grid - number of blocks
  const size_t reductionGrid  = (red_thr + red_thr_blc - 1) / red_thr_blc;

  // Log benchmark setup
  printf("N: %d\n", N);
  printf("dt: %f\n", dt);
  printf("steps: %d\n", steps);
  printf("threads/block: %d\n", thr_blc);
  printf("blocks/grid: %lu\n", simulationGrid);
  printf("reduction threads/block: %d\n", red_thr_blc);
  printf("reduction blocks/grid: %lu\n", reductionGrid);

  const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;
  writeFreq = (writeFreq > 0) ?  writeFreq : 0;


  t_particles particles_cpu;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                            FILL IN: CPU side memory allocation (step 0)                                          //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  particles_cpu.pos_x = (float *)malloc(N * sizeof(float));
  particles_cpu.pos_y = (float *)malloc(N * sizeof(float));
  particles_cpu.pos_z = (float *)malloc(N * sizeof(float));
  particles_cpu.vel_x = (float *)malloc(N * sizeof(float));
  particles_cpu.vel_y = (float *)malloc(N * sizeof(float));
  particles_cpu.vel_z = (float *)malloc(N * sizeof(float));
  particles_cpu.weight = (float *)malloc(N  * sizeof(float));


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                              FILL IN: memory layout descriptor (step 0)                                          //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                      Stride of two               Offset of the first
   *  Data pointer        consecutive elements        element in floats,
   *                      in floats, not bytes        not bytes
  */
  MemDesc md(
        particles_cpu.pos_x ,                1,                          0,              // Postition in X
        particles_cpu.pos_y ,                1,                          0,              // Postition in Y
        particles_cpu.pos_z ,                1,                          0,              // Postition in Z
        particles_cpu.vel_x ,                1,                          0,              // Velocity in X
        particles_cpu.vel_y ,                1,                          0,              // Velocity in Y
        particles_cpu.vel_z ,                1,                          0,              // Velocity in Z
        particles_cpu.weight,                1,                          0,              // Weight
        N,                                                                  // Number of particles
        recordsNum);                                                        // Number of records in output file

  // Initialisation of helper class and loading of input data
  H5Helper h5Helper(argv[8], argv[9], md);

  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  }
  catch (const std::exception& e)
  {
    std::cerr<<e.what()<<std::endl;
    return -1;
  }


  t_particles particles_gpu_in;
  t_particles particles_gpu_out;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                  FILL IN: GPU side memory allocation (step 1)                                    //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // p_in
  cudaMalloc<float>(&particles_gpu_in.pos_x, N * sizeof(float));
  cudaMalloc<float>(&particles_gpu_in.pos_y, N * sizeof(float));
  cudaMalloc<float>(&particles_gpu_in.pos_z, N * sizeof(float));

  cudaMalloc<float>(&particles_gpu_in.vel_x, N * sizeof(float));
  cudaMalloc<float>(&particles_gpu_in.vel_y, N * sizeof(float));
  cudaMalloc<float>(&particles_gpu_in.vel_z, N * sizeof(float));

  cudaMalloc<float>(&particles_gpu_in.weight, N * sizeof(float));

  // p_out
  cudaMalloc<float>(&particles_gpu_out.pos_x, N * sizeof(float));
  cudaMalloc<float>(&particles_gpu_out.pos_y, N * sizeof(float));
  cudaMalloc<float>(&particles_gpu_out.pos_z, N * sizeof(float));

  cudaMalloc<float>(&particles_gpu_out.vel_x, N * sizeof(float));
  cudaMalloc<float>(&particles_gpu_out.vel_y, N * sizeof(float));
  cudaMalloc<float>(&particles_gpu_out.vel_z, N * sizeof(float));

  cudaMalloc<float>(&particles_gpu_out.weight, N * sizeof(float));


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                       FILL IN: memory transfers (step 1)                                         //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  cudaMemcpy(particles_gpu_in.pos_x, particles_cpu.pos_x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpu_in.pos_y, particles_cpu.pos_y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpu_in.pos_z, particles_cpu.pos_z, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(particles_gpu_in.vel_x, particles_cpu.vel_x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpu_in.vel_y, particles_cpu.vel_y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpu_in.vel_z, particles_cpu.vel_z, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(particles_gpu_in.weight, particles_cpu.weight, N * sizeof(float), cudaMemcpyHostToDevice);


  cudaMemcpy(particles_gpu_out.pos_x, particles_cpu.pos_x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpu_out.pos_y, particles_cpu.pos_y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpu_out.pos_z, particles_cpu.pos_z, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(particles_gpu_out.vel_x, particles_cpu.vel_x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpu_out.vel_y, particles_cpu.vel_y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpu_out.vel_z, particles_cpu.vel_z, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(particles_gpu_out.weight, particles_cpu.weight, N * sizeof(float), cudaMemcpyHostToDevice);
  

  gettimeofday(&t1, 0);

  dim3 blockDim(thr_blc);
  dim3 gridDim(simulationGrid);

  t_particles particles_gpu_tmp;

  for(int s = 0; s < steps; s++)
  {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       FILL IN: kernels invocation (step 1)                                     //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    calculate_velocity<<<gridDim, blockDim>>>(particles_gpu_in, particles_gpu_out, N, dt);
    particles_gpu_tmp = particles_gpu_in;
    particles_gpu_in = particles_gpu_out;
    particles_gpu_out = particles_gpu_tmp;
    

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                          FILL IN: synchronization  (step 4)                                    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (writeFreq > 0 && (s % writeFreq == 0))
    {
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //                          FILL IN: synchronization and file access logic (step 4)                             //
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //              FILL IN: invocation of center-of-mass kernel (step 3.1, step 3.2, step 4)                           //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  float *com_x;
  float *com_y;
  float *com_z;
  float *com_w;
  cudaMalloc<float>(&com_x, sizeof(float)); 
  cudaMalloc<float>(&com_y, sizeof(float)); 
  cudaMalloc<float>(&com_z, sizeof(float)); 
  cudaMalloc<float>(&com_w, sizeof(float)); 

  cudaMemset(com_x, 0, sizeof(float));
  cudaMemset(com_y, 0, sizeof(float));
  cudaMemset(com_z, 0, sizeof(float));
  cudaMemset(com_w, 0, sizeof(float));

  int* lock;
  cudaMallocManaged((void**)&lock, sizeof(int));
  *lock = 0;

  dim3 blockDimRed(red_thr_blc);
  dim3 gridDimRed(reductionGrid);

  centerOfMass<<<gridDimRed, blockDimRed, (red_thr_blc * sizeof(float) * 4) / 32>>>
                    (particles_gpu_in, com_x, com_y, com_z, com_w, lock, N);

  //
  cudaDeviceSynchronize();

  gettimeofday(&t2, 0);

  // Approximate simulation wall time
  double t = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
  printf("Time: %f s\n", t);


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                             FILL IN: memory transfers for particle data (step 1)                                 //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  cudaMemcpy(particles_cpu.pos_x, particles_gpu_in.pos_x, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.pos_y, particles_gpu_in.pos_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.pos_z, particles_gpu_in.pos_z, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaMemcpy(particles_cpu.vel_x, particles_gpu_in.vel_x, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.vel_y, particles_gpu_in.vel_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.vel_z, particles_gpu_in.vel_z, N * sizeof(float), cudaMemcpyDeviceToHost);
  
  float4 comOnGPU;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                        FILL IN: memory transfers for center-of-mass (step 3.1, step 3.2)                         //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  cudaMemcpy(&comOnGPU.x, com_x, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&comOnGPU.y, com_y, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&comOnGPU.z, com_z, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&comOnGPU.w, com_w, sizeof(float), cudaMemcpyDeviceToHost);

  float4 comOnCPU = centerOfMassCPU(md);

  std::cout << "Center of mass on CPU:" << std::endl
            << comOnCPU.x <<", "
            << comOnCPU.y <<", "
            << comOnCPU.z <<", "
            << comOnCPU.w
            << std::endl;

  std::cout << "Center of mass on GPU:" << std::endl
            << comOnGPU.x<<", "
            << comOnGPU.y<<", "
            << comOnGPU.z<<", "
            << comOnGPU.w
            << std::endl;

  // Writing final values to the file
  h5Helper.writeComFinal(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w);
  h5Helper.writeParticleDataFinal();

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
