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


__global__ void calculate_velocity(t_particles p_in, t_particles p_out, int N, float dt){

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if ( idx >= N )
    return;

  float r, dx, dy, dz;
  float vx, vy, vz;

  float tmp_vel_x = 0, tmp_vel_y = 0, tmp_vel_z = 0;

  float r3, G_dt_r3, Fg_dt_m2_r;

  // each thread will take one particle and compare it with the rest
  for (int i = 0; i < N; i++){
    // Same For both
    dx = p_in.pos_x[idx] - p_in.pos_x[i];
    dy = p_in.pos_y[idx] - p_in.pos_y[i];
    dz = p_in.pos_z[idx] - p_in.pos_z[i];

    // non coliding velocities
    r = sqrt(dx*dx + dy*dy + dz*dz);

    r3 = r * r * r + FLT_MIN;

    G_dt_r3 = -G * dt / r3;
    Fg_dt_m2_r = G_dt_r3 * p_in.weight[i];

    vx = Fg_dt_m2_r * dx;
    vy = Fg_dt_m2_r * dy;
    vz = Fg_dt_m2_r * dz;

    // non coliding velocity
    tmp_vel_x += (r > COLLISION_DISTANCE) ? vx : 0.0f;
    tmp_vel_y += (r > COLLISION_DISTANCE) ? vy : 0.0f;
    tmp_vel_z += (r > COLLISION_DISTANCE) ? vz : 0.0f;


    // coliding velocities
    vx = ((p_in.weight[idx] * p_in.vel_x[idx] - p_in.weight[i] * p_in.vel_x[idx] + 2 * p_in.weight[i] * p_in.vel_x[i]) /
      (p_in.weight[idx] + p_in.weight[i])) - p_in.vel_x[idx];
    vy = ((p_in.weight[idx] * p_in.vel_y[idx] - p_in.weight[i] * p_in.vel_y[idx] + 2 * p_in.weight[i] * p_in.vel_y[i]) /
      (p_in.weight[idx] + p_in.weight[i])) - p_in.vel_y[idx];
    vz = ((p_in.weight[idx] * p_in.vel_z[idx] - p_in.weight[i] * p_in.vel_z[idx] + 2 * p_in.weight[i] * p_in.vel_z[i]) /
      (p_in.weight[idx] + p_in.weight[i])) - p_in.vel_z[idx];

    if (r > 0.0f && r < COLLISION_DISTANCE) {
      tmp_vel_x += vx;
      tmp_vel_y += vy;
      tmp_vel_z += vz;
    }
  }

  // update particle
  p_out.vel_x[idx] = p_in.vel_x[idx] + tmp_vel_x;
  p_out.vel_y[idx] = p_in.vel_y[idx] + tmp_vel_y;
  p_out.vel_z[idx] = p_in.vel_z[idx] + tmp_vel_z;
  
  p_out.pos_x[idx] = p_in.pos_x[idx] + p_out.vel_x[idx] * dt;
  p_out.pos_y[idx] = p_in.pos_y[idx] + p_out.vel_y[idx] * dt;
  p_out.pos_z[idx] = p_in.pos_z[idx] + p_out.vel_z[idx] * dt;
}


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
