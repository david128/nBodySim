#include "implementKernel.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>


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

			if (i != j)
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

		float vdt[3] = { pArray[i].velocity.x * timeStep,pArray[i].velocity.y * timeStep,pArray[i].velocity.z * timeStep };

		pArray[i].nextPosition.x = pArray[i].position.x + vdt[0];
		pArray[i].nextPosition.y = pArray[i].position.y + vdt[1];
		pArray[i].nextPosition.z = pArray[i].position.z + vdt[2];
	}

}



//__global__
//void buildTreeSplit(Level prevLevel, NodeGPU* nodes)
//{
//
//	int threadID = threadIdx.x + blockIdx.x * blockDim.x;  //index of body based on thread 
//	int stride = blockDim.x * gridDim.x; //stride legnth based on number of threads
//	int numOfNodes = (prevLevel.treeLevel + 1) * 8;
//	int nodeIndex = threadID + prevLevel.minIndex;
//
//	while (nodeIndex < numOfNodes) 
//	{
//		NodeGPU &currentNode= nodes[nodeIndex];
//		//find direction centre point of current node
//		float halfSide = currentNode.sideLegnth * 0.5;
//		//float parentCentreX = currentNode.position.x - halfSide;
//		//float parentCentreY = currentNode.position.y - halfSide;
//		//float parentCentreZ = currentNode.position.z - halfSide;
//
//		//if (parentCentreX == 0.0f || parentCentreY == 0.0f || parentCentreZ == 0.0f)
//		//{
//		//	parentCentreX = 0.01f;//alter to avoid dividing by 0
//		//	parentCentreY = 0.01f;
//		//	parentCentreZ = 0.01f;
//		//}
//
//		//create 8 nodes
//		for (int i = 0; i < 8; i++)
//		{
//			currentNode.children[i] = new NodeGPU();
//
//		}
//
//
//		
//		//allocate particles
//		//end
//		
//	}
//}

__global__ void rootKernel(int* child, int numNodes)
{
	int k = numNodes;
	k *= 8;
	int f = 0;
	for (int i = 0; i < 8; i++)
	{
		child[k + i] = -1;
		
		f = child[k + i];

		int s = f;

	}
}


__global__ void  clearKernel(int* child, int numNodes, int n)
{
	int k, stride, end;

	end = 8 * numNodes;
	
	stride = blockDim.x * gridDim.x;
	k = 0;


	// iterate cells and reset to 0;
	while (k < end) 
	{
		child[k] = -1;
		k += stride;
	}
}

__global__ void buildTreeInsertion(NodeGPU* root, int n, Particle* pArray, int *child, int numNodes)
{
	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;  //index of body based on thread 
	int stride = blockDim.x * gridDim.x; //stride legnth based on number of threads
	bool skip = false;
	int currentChild;
	int temp = 0;
	int depth;
	int locked;
	int patch;
	bool success = false;

	float nodeX;
	float nodeY;
	float nodeZ;
	float dx, dy, dz;
	float childHalfSide;
	int childIndex = 0;
	int cell;
	int bottom = numNodes;
	float bx, by, bz;
	float rootHalfSide = root->sideLegnth * 0.5f;
	float rootCx = root->position.x - rootHalfSide;
	float rootCy = root->position.y - rootHalfSide;
	float rootCz = root->position.z - rootHalfSide;

	while (bodyIndex < n)
	{

		if (!skip)//new body 
		{
			{
				bx = pArray[bodyIndex].position.x;
				by = pArray[bodyIndex].position.y;
				bz = pArray[bodyIndex].position.z;
			}

			skip = true;
			success = false;
			depth = 1;
			//find child index
			

			childIndex = 0;
			temp = numNodes;

			childHalfSide = rootHalfSide * 0.5f;
			dx = dy = dz = -childHalfSide;
			
			if (rootCx < bx)
			{
				childIndex = 1;
				dx = childHalfSide;
			}
			if (rootCy < by)
			{
				childIndex |= 2;
				dy = childHalfSide;
			}
			if (rootCz < bz)
			{
				childIndex |= 4;
				dz = childHalfSide;
			}
			nodeX = rootCx + dx;
			nodeY = rootCy + dy;
			nodeZ = rootCz + dz;

		}

		//go to leaf node
		currentChild = child[temp * 8 + childIndex];

		while (currentChild >= n)
		{
			

			temp = currentChild;
			depth++;

			childHalfSide = rootHalfSide * 0.5f;
			dx = dy = dz = -childHalfSide;

			//find child index

			if (nodeX < bx)
			{
				childIndex = 1;
				dx = childHalfSide;
			}
			if (nodeY < by)
			{
				childIndex |= 2;
				dy = childHalfSide;
			}
			if (nodeZ < bz)
			{
				childIndex |= 4;
				dz = childHalfSide;
			}
			nodeX +=  dx;
			nodeY +=  dy;
			nodeZ +=  dz;
			currentChild = child[temp * 8 + childIndex];
		}

		if (currentChild != -2)
		{
			locked = temp * 8 + childIndex; //store index
			if (currentChild == -1)
			{
				if (-1 == atomicCAS((int*)&child[locked], -1, bodyIndex))//if null insert body
				{
					bodyIndex += stride;
					skip = false;
				}
			}
			else
			{
				
				if (currentChild == atomicCAS((int*)&child[locked], currentChild, -2)) //already a body here, need to split.
				{	
					patch = -1;
					do
					{
						depth++;


						if (depth > 50)
						{
							
							float s;
							s = bx;
							s = by;
							s = bz;
						}

						cell = atomicSub((int*) &bottom, 1) - 1;

						if (patch != -1) 
						{
							child[temp * 8 + childIndex] = cell;
						}

						if (patch < cell)
						{
							patch = cell;
						}

						childIndex = 0;
						float cx, cy, cz;

						{
							cx = pArray[currentChild].position.x;
							cy = pArray[currentChild].position.y;
							cz = pArray[currentChild].position.z;
						}

						if (nodeX < cx) childIndex = 1;
						if (nodeY < cy) childIndex |= 2;
						if (nodeZ < cz) childIndex |= 4;
						child[cell * 8 + childIndex] = currentChild; //inserting old body inro child cell
					
						temp = cell;
						childHalfSide = childHalfSide * 0.5f;
						dx = dy = dz = -childHalfSide;
						childIndex = 0;
						if (nodeX < bx)
						{
							childIndex = 1;
							dx = childHalfSide;
						}
						if (nodeY < by) 
						{
							childIndex |= 2;
							dy = childHalfSide;
						}
						if (nodeZ < bz)
						{
							childIndex |= 4;
							dz = childHalfSide;
						}
						nodeX += dx;
						nodeY += dy;
						nodeZ += dz;

						currentChild = child[temp * 8 + childIndex]; //current child cell = new body's child if the same as old then ch will be equal to old body index and therefore !>= 0, so loop continues
					} while (currentChild >= 0); //repeat until bodies in differnet children
					child[temp*8 + childIndex] = bodyIndex;//insert new body
					bodyIndex += stride;
					success = true;
					skip = false;
				}
			}
		}
		__syncthreads();

		if (success)
		{
			child[locked] = patch;
		}
	}
}
