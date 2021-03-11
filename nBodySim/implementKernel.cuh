#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "ParticleManager.h"


__global__
void split(int* f)
{

}

//n = number of bodies
//pp =particle position(x/y/z)
//pm = particle mass
__global__
void AllPairs(unsigned int n, float* ppx,float* ppy, float* ppz, float* pm, float timeStep)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int numThreads = blockDim.x * gridDim.x;

	float g = 6.67408e-11f; //grav constant

	for (int i = id; i < 100; i += numThreads) //this will loop in i, incrementing by number of threads in parallel
	{
		float acc[3] = { 0.0f,0.0f,0.0f };
		for (int j = 0; j < 100; j++) //j loop that increments by 1 calculating acc in serial
		{
		
			float diff[3] = { ppx[i] - ppx[j], ppy[i] - ppy[j], ppz[i] - ppz[j] };
			float dist = sqrtf(diff[0]*diff[0] + diff[1] * diff[1]+ diff[2] * diff[2]); //get distance
			float multiplier = (g * pm[j]) / (dist * dist * dist); //multiplier  (g * mass )/ (distance ^3)


			diff[0] = diff[0] * -multiplier;
			diff[1] = diff[1] * -multiplier;
			diff[2] = diff[2] * -multiplier;

			acc[0] += diff[0];
			acc[1] += diff[1];
			acc[2] += diff[2];
			
		}
		//v(t+1) = v(t) +a(t) *dt
		acc[0] *= timeStep;
		acc[1] *= timeStep;
		acc[2] *= timeStep;
		//particles->at(i)->velocity = particles->at(i)->velocity + acc;

		////x(t+1) = x(t) + v(t)*dt
		//Vector3 vDt = particles->at(i)->velocity;
		//vDt.scale(timeStep);
		//particles->at(i)->nextPosition = particles->at(i)->position + vDt;
	}

}
//
//__global__ void KernelcomputeForces(unsigned int n, float* gm, float* gpx, float* gpy, float* gpz, float* gfx, float* gfy, float* gfz) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	int numThreads = blockDim.x * gridDim.x;
//
//	float GRAVITY = 0.00001f;
//
//	//compare all with all
//	for (unsigned int ia = tid; ia < n; ia += numThreads) {
//		float lfx = 0.0f;
//		float lfy = 0.0f;
//		float lfz = 0.0f;
//
//		for (unsigned int ib = 0; ib < n; ib++) {
//			//compute distance
//			float dx = (gpx[ib] - gpx[ia]);
//			float dy = (gpy[ib] - gpy[ia]);
//			float dz = (gpz[ib] - gpz[ia]);
//			//float distance = sqrt( dx*dx + dy*dy + dz*dz );
//			float distanceSquared = dx * dx + dy * dy + dz * dz;
//
//			//prevent slingshots and division by zero
//			//distance += 0.1f;
//			distanceSquared += 0.01f;
//
//			//calculate gravitational magnitude between the bodies
//			//float magnitude = GRAVITY * ( gm[ia] * gm[ib] ) / ( distance * distance * distance * distance );
//			float magnitude = GRAVITY * (gm[ia] * gm[ib]) / (distanceSquared);
//
//			//calculate forces for the bodies
//			//magnitude times direction
//			lfx += magnitude * (dx);
//			lfy += magnitude * (dy);
//			lfz += magnitude * (dz);
//		}
//
//		//stores local memory to global memory
//		gfx[ia] = lfx;
//		gfy[ia] = lfy;
//		gfz[ia] = lfz;
//	}
//}

//extern void GPUcomputeForces(unsigned int n, float* gm, float* gpx, float* gpy, float* gpz, float* gfx, float* gfy, float* gfz) {
//	dim3 gridDim(16, 1, 1); //specifys how many blocks in three possible dimensions
//	dim3 blockDim(512, 1, 1); //threads per block
//	KernelcomputeForces << <gridDim, blockDim >> > (n, gm, gpx, gpy, gpz, gfx, gfy, gfz);
//}