#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "Particle.h"
#include "BarnesHut.cuh"





//n = number of bodies
__global__
void AllPairs(unsigned int n, Particle* pArray, float timeStep)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x; //id of current thread
	int stride = blockDim.x * gridDim.x; //used to stride by number of threads

	float g = 6.67408e-11f; //grav constant

	for (int i = id; i < n; i += stride) //this will loop in i, incrementing by number of threads in parallel
	{
		float acc[3] = { 0.0f,0.0f,0.0f };
		for (int j = 0; j < n; j++) //j loop that increments by 1 calculating acc in serial
		{
		
			if (i!=j)
			{
				float diff[3] = { pArray[i].position.x - pArray[j].position.x, pArray[i].position.y - pArray[j].position.y, pArray[i].position.z - pArray[j].position.z };
				float dist = sqrtf(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]); //get distance
				float multiplier = (g * pArray[j].mass) / (dist * dist * dist); //multiplier  (g * mass )/ (distance ^3)


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

		float vdt[3] = { pArray[i].velocity.x  * timeStep,pArray[i].velocity.y * timeStep,pArray[i].velocity.z * timeStep };

		pArray[i].nextPosition.x= pArray[i].position.x + vdt[0];
		pArray[i].nextPosition.y= pArray[i].position.y + vdt[1];
		pArray[i].nextPosition.z= pArray[i].position.z + vdt[2];
	}

}

__global__
void buildTree(unsigned int n, Particle* pArray, NodeGPU* root)
{
	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;  //index of body based on thread 
	int stride = blockDim.x * gridDim.x; //stride legnth based on number of threads

	

	//traverse tree until get to leaf node,

	NodeGPU* currentNode = root;
	
	while (bodyIndex < n)
	{
		if (currentNode->particleCount == 0)
		{
			//insert particle
			currentNode->particles[0] = pArray[bodyIndex];
			currentNode->particleCount++;

		}
		else if(currentNode->particleCount > 0)
		{
			//find direction centre point of current node
			float halfSide = currentNode->sideLegnth * 0.5;
			float parentCentreX = currentNode->position.x - halfSide;
			float parentCentreY = currentNode->position.y - halfSide;
			float parentCentreZ = currentNode->position.z - halfSide;

			if (parentCentreX == 0.0f || parentCentreY == 0.0f || parentCentreZ == 0.0f)
			{
				parentCentreX = 0.01f;//alter to avoid dividing by 0
				parentCentreY = 0.01f;
				parentCentreZ = 0.01f;
			}

			float dirX = pArray[bodyIndex].position.x - parentCentreX;
			float dirY = pArray[bodyIndex].position.y - parentCentreY;
			float dirZ = pArray[bodyIndex].position.z - parentCentreZ;

			dirX = dirX / abs(dirX);
			dirY = dirY / abs(dirY);
			dirZ = dirZ / abs(dirZ);

			int childIndex = 0;

			if (dirX < 0.0) //we have halfed the octant now, so will either be index (0,1,2,3) or (4,5,6,7)
			{
				childIndex = 4;
			}
			if (dirY < 0.0) //half it again from  first one, if in (0,1,2,3) section on X if dirY >0 then will have possible values of (0/1) if dirY < 0 then possible (2/3)
			{
				childIndex = childIndex + 2;
			}
			if (dirZ < 0.0) //final split, if childIndex is still 0 then value will be either 0 or 1
			{
				childIndex = childIndex + 1;
			}

			bool inserted = false;
			while (!inserted) //
			{
				//check if cell is locked
				if (!currentNode->children[childIndex]->locked)
				{
					//now insert child
					
					
				}
			}

		}
	}

}


