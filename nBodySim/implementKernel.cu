#include "implementKernel.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>


__global__ void EulerAcceleration(unsigned int n, Particle* pArray, float timeStep)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x; //id of current thread
	int stride = blockDim.x * gridDim.x; //used to stride by number of threads

	float g = 6.67408e-11f; //grav constant

	float acc[3] = { 0.0f,0.0f,0.0f };
	float diff[3];
	float dist;
	float multiplier;

	for (int i = id; i < n; i += stride) //this will loop in i, incrementing by number of threads in parallel
	{
		acc[0] = 0.0f;
		acc[1] = 0.0f;
		acc[2] = 0.0f;
		for (int j = 0; j < n; j++) //j loop that increments by 1 calculating acc in serial
		{

			if (i != j)
			{
				diff[0] = pArray[i].position.x - pArray[j].position.x;
				diff[1] = pArray[i].position.y - pArray[j].position.y;
				diff[2] = pArray[i].position.z - pArray[j].position.z;
				dist = sqrtf(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]); //get distance
				multiplier = (g * pArray[j].mass) / (dist * dist * dist); //multiplier  (g * mass )/ (distance ^3)


				diff[0] = diff[0] * -multiplier;
				diff[1] = diff[1] * -multiplier;
				diff[2] = diff[2] * -multiplier;

				acc[0] += diff[0];
				acc[1] += diff[1];
				acc[2] += diff[2];
			}



		}
		//v(t+1) = v(t) +a(t) *dt
		acc[0] *= timeStep;
		acc[1] *= timeStep;
		acc[2] *= timeStep;

		pArray[i].acceleration.x = acc[0];
		pArray[i].acceleration.y = acc[1];
		pArray[i].acceleration.z = acc[2];

		pArray[i].velocity.x += acc[0];
		pArray[i].velocity.y += acc[1];
		pArray[i].velocity.z += acc[2];

	}

}

__global__ void EulerPosition(unsigned int n, Particle* pArray, float timeStep)
{

	int id = blockDim.x * blockIdx.x + threadIdx.x; //id of current thread
	int stride = blockDim.x * gridDim.x; //used to stride by number of threads
	float vdt[3];
	for (int i = id; i < n; i += stride)
	{
		vdt[0] = (pArray[i].velocity.x * timeStep);
		vdt[1] = (pArray[i].velocity.y * timeStep);
		vdt[2] = (pArray[i].velocity.z * timeStep );
		pArray[i].position.x += vdt[0];
		pArray[i].position.y += vdt[1];
		pArray[i].position.z += vdt[2];
	}
	


}
