/**
 * @File nbody.cu
 *
 * Implementation of the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2020)
 * Projekt c. 1 (cuda)
 * Login: xlukac11
 */

#include <cmath>
#include <cfloat>
#include "nbody.h"

/**
 * CUDA kernel to calculate gravitation velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_gravitation_velocity(t_particles p, t_velocities tmp_vel, int N, float dt)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if ( idx >= N )
    return;
  
  float r, dx, dy, dz;
  float F;
  float nx, ny, nz;
  float vx, vy, vz;

  // each thread will take one particle and compare it with the rest
  for (int i = 0; i < N; i++){
    dx = p.pos_x[idx] - p.pos_x[i];
    dy = p.pos_y[idx] - p.pos_y[i];
    dz = p.pos_z[idx] - p.pos_z[i];

    r = sqrt(dx*dx + dy*dy + dz*dz);

    F = -G * p.weight[idx] * p.weight[i] / (r * r + FLT_MIN);

    nx = F * dx/ (r + FLT_MIN);
    ny = F * dy/ (r + FLT_MIN);
    nz = F * dz/ (r + FLT_MIN);

    vx = nx * dt / p.weight[idx];
    vy = ny * dt / p.weight[idx];
    vz = nz * dt / p.weight[idx];

    tmp_vel.x[idx] += (r > COLLISION_DISTANCE) ? vx : 0.0f;
    tmp_vel.y[idx] += (r > COLLISION_DISTANCE) ? vy : 0.0f;
    tmp_vel.z[idx] += (r > COLLISION_DISTANCE) ? vz : 0.0f;
  }
  
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_collision_velocity(t_particles p, t_velocities tmp_vel, int N, float dt)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if ( idx >= N )
    return;

  float r, dx, dy, dz;
  float vx, vy, vz;

  // each thread will take one particle and compare it with the rest
  for (int i = 0; i < N; i++){
    dx = p.pos_x[idx] - p.pos_x[i];
    dy = p.pos_y[idx] - p.pos_y[i];
    dz = p.pos_z[idx] - p.pos_z[i];

    r = sqrtf(dx*dx + dy*dy + dz*dz);

    /* MISTO PRO VAS KOD KOLIZE */

    vx = ((p.weight[idx] * p.vel_x[idx] - p.weight[i] * p.vel_x[idx] + 2 * p.weight[i] * p.vel_x[i]) /
            (p.weight[idx] + p.weight[i])) - p.vel_x[idx] ;
    vy = ((p.weight[idx] * p.vel_y[idx] - p.weight[i] * p.vel_y[idx] + 2 * p.weight[i] * p.vel_y[i]) /
            (p.weight[idx] + p.weight[i])) - p.vel_y[idx] ;
    vz = ((p.weight[idx] * p.vel_z[idx] - p.weight[i] * p.vel_z[idx] + 2 * p.weight[i] * p.vel_z[i]) /
            (p.weight[idx] + p.weight[i])) - p.vel_z[idx] ;


    /* KONEC */

    // jedna se o rozdilne ale blizke prvky
    if (r > 0.0f && r < COLLISION_DISTANCE) {
        tmp_vel.x[idx] += vx;
        tmp_vel.y[idx] += vy;
        tmp_vel.z[idx] += vz;
    }
  }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void update_particle(t_particles p, t_velocities tmp_vel, int N, float dt)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if ( idx >= N )
    return;

  p.vel_x[idx] += tmp_vel.x[idx];
  p.vel_y[idx] += tmp_vel.y[idx];
  p.vel_z[idx] += tmp_vel.z[idx];

  p.pos_x[idx] += p.vel_x[idx] * dt;
  p.pos_y[idx] += p.vel_y[idx] * dt;
  p.pos_z[idx] += p.vel_z[idx] * dt;

}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param comX    - pointer to a center of mass position in X
 * @param comY    - pointer to a center of mass position in Y
 * @param comZ    - pointer to a center of mass position in Z
 * @param comW    - pointer to a center of mass weight
 * @param lock    - pointer to a user-implemented lock
 * @param N       - Number of particles
 */
__global__ void centerOfMass(t_particles p, float* comX, float* comY, float* comZ, float* comW, int* lock, const int N)
{

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassCPU(MemDesc& memDesc)
{
  float4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting points and most recent position of center-of-mass
    const float dx = memDesc.getPosX(i) - com.x;
    const float dy = memDesc.getPosY(i) - com.y;
    const float dz = memDesc.getPosZ(i) - com.z;

    // Calculate weight ratio only if at least one particle isn't massless
    const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                          ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += dx * dw;
    com.y += dy * dw;
    com.z += dz * dw;
    com.w += memDesc.getWeight(i);
  }
  return com;
}// enf of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
