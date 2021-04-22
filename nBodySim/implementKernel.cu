#include "implementKernel.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>


__device__ int bottom;

const int MAX_N = 1000;

//n = number of bodies
__global__
void EulerAcceleration(unsigned int n, Particle* pArray, float timeStep)
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

	}

}

__global__ void EulerPosition(unsigned int n, Particle* pArray, float timeStep)
{

	int id = blockDim.x * blockIdx.x + threadIdx.x; //id of current thread
	int stride = blockDim.x * gridDim.x; //used to stride by number of threads
	
	for (int i = id; i < n; i += stride)
	{
		float vdt[3] = { pArray[i].velocity.x * timeStep, pArray[i].velocity.y * timeStep,pArray[i].velocity.z * timeStep };
		pArray[i].position.x += vdt[0];
		pArray[i].position.y += vdt[1];
		pArray[i].position.z += vdt[2];
	}
	


}


__global__ void rootKernel(int* child, float* mass, int numNodes)
{
	int index = numNodes;
	int f = 0;
	bottom = numNodes;
	mass[index] = -1.0f;

	index *= 8;
	for (int i = 0; i < 8; i++)
	{
		child[index + i] = -1;
		


	}
}


__global__ void  clearKernel(int* child, float* mass, int numNodes, int n)
{
	int index, stride, end;

	end = 8 * numNodes;
	
	stride = blockDim.x * gridDim.x;
	index = threadIdx.x + blockIdx.x * blockDim.x; ;


	// iterate cells and reset to 0;
	while (index < end) 
	{
		child[index] = -1;
		mass[index] = -1.0f;
		index += stride;
	}
}

__global__ void buildTreeInsertion(NodeGPU* root, int n, Particle* pArray, int *child, int numNodes)
{
	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;  //index of body based on thread 
	int stride = blockDim.x * gridDim.x; //stride legnth based on number of threads
	int skip;
	int currentChild;
	int temp;
	int depth;
	int locked;
	int patch;
	bool success;

	float nodeX;
	float nodeY;
	float nodeZ;
	float dx, dy, dz;
	float childHalfSide;
	int childIndex;
	int cell;
	
	float bx, by, bz;
	float rootHalfSide = root->sideLegnth * 0.5f;
	float rootCx = 0.0f ;
	float rootCy = 0.0f ;
	float rootCz = 0.0f ;


	int times = 0;
	bool b;

	skip = 1;
	success = false;

	while (bodyIndex < n)
	{

		times++;
		if (times > 10000)
		{
			b = true;
		}

		{
			bx = pArray[bodyIndex].position.x;
			by = pArray[bodyIndex].position.y;
			bz = pArray[bodyIndex].position.z;
		}

		if (skip !=0)//new body 
		{


			skip = 0;
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
			

			//count this node, add mass, work out centre of mass
			//atomicAdd((float*)&mass[temp * 8 + childIndex], pArray[bodyIndex].mass);//adds mass
			//atomicAdd((int*)&count[temp * 8 + childIndex], 1);
			//int newCount = count[temp * 8 + childIndex];
			//int oldCount = newCount - 1;

			////centre of mass
			////new cm = old count * (cm/new count) + new point/ new count
			//float newCMX = oldCount * (cmx[temp * 8 + childIndex] / newCount) + bx / newCount;
			//float newCMY = oldCount * (cmy[temp * 8 + childIndex] / newCount) + by / newCount;
			//float newCMZ = oldCount * (cmz[temp * 8 + childIndex] / newCount) + bz / newCount;

			//atomicExch((float*)&cmx[temp * 8 + childIndex], newCMX);
			//atomicExch((float*)&cmy[temp * 8 + childIndex], newCMY);
			//atomicExch((float*)&cmz[temp * 8 + childIndex], newCMZ);

			temp = currentChild;

			depth++;

			childHalfSide = rootHalfSide * 0.5f;
			dx = dy = dz = -childHalfSide;

			//find child index
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

					////adds mass
					//atomicAdd((float*)&mass[locked], pArray[bodyIndex].mass);

					////adds centre of mass
					//atomicAdd((float*)&cmx[locked], bx);
					//atomicAdd((float*)&cmy[locked], by);
					//atomicAdd((float*)&cmz[locked], bz);
					////counts
					//atomicAdd((int*)&count[locked], 1);
					bodyIndex += stride;
					skip = 1;
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

						
						cx = pArray[currentChild].position.x;
						cy = pArray[currentChild].position.y;
						cz = pArray[currentChild].position.z;
						

						if (nodeX < cx) childIndex = 1;
						if (nodeY < cy) childIndex |= 2;
						if (nodeZ < cz) childIndex |= 4;
						child[cell * 8 + childIndex] = currentChild; //inserting old body into child cell

						//atomicAdd((float*)&mass[cell * 8 + childIndex], pArray[bodyIndex].mass);//adds mass
						//atomicAdd((int*)&count[cell * 8 + childIndex], 1);
						//int newCount = count[cell * 8 + childIndex];
						//int oldCount = newCount - 1;

						//adds centre of mass
						//new cm = old count * (cm/new count) + new point/ new count
						/*float newCMX = oldCount * (cmx[cell * 8 + childIndex] / newCount) + cx / newCount;
						float newCMY = oldCount * (cmy[cell * 8 + childIndex] / newCount) + cy / newCount;
						float newCMZ = oldCount * (cmz[cell * 8 + childIndex] / newCount) + cz / newCount;

						atomicExch((float*)&cmx[cell * 8 + childIndex], newCMX);
						atomicExch((float*)&cmy[cell * 8 + childIndex], newCMY);
						atomicExch((float*)&cmz[cell * 8 + childIndex], newCMZ);*/
					
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

						//atomicAdd((float*)&mass[temp * 8 + childIndex], pArray[bodyIndex].mass);//adds mass
						//atomicAdd((int*)&count[temp * 8 + childIndex], 1);
						//newCount = count[temp * 8 + childIndex];
						//oldCount = newCount - 1;

						////adds centre of mass
						////new cm = old count * (cm/new count) + new point/ new count
						//newCMX = oldCount * (cmx[temp * 8 + childIndex] / newCount) + bx / newCount;
						//newCMY = oldCount * (cmy[temp * 8 + childIndex] / newCount) + by / newCount;
						//newCMZ = oldCount * (cmz[temp * 8 + childIndex] / newCount) + bz / newCount;

						//atomicExch((float*)&cmx[temp * 8 + childIndex], newCMX);
						//atomicExch((float*)&cmy[temp * 8 + childIndex], newCMY);
						//atomicExch((float*)&cmz[temp * 8 + childIndex], newCMZ);

					} while (currentChild >= 0); //repeat until bodies in differnet children
					child[temp*8 + childIndex] = bodyIndex;//insert new body

					

					bodyIndex += stride;
					success = true;
					skip = 2;

					__threadfence();

					if (skip == 2)
					{
						1 * 3;
					}
					else
					{
						1 * 2;
					}

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

__global__ void SumKernel(int* child, int* count, int numNodes, int n,  float* mass, Particle* pArray)
{
	
	
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int currentChild;
	int i = 0;

	int tempChild[8];//probably should be more
	float tempMass[8];

	bool success;

	float cm, m, px,py,pz;
	int counter;

	if (index < bottom) index = index + stride;

	int restart = index;

	for (int j= 0; j < 5; j++)
	{
		while (index <= numNodes)
		{
			if (mass[index] < 0.0f)
			{
				for (i = 0; i < 8; i++)
				{
					currentChild = child[index * 8 + i];
					tempChild[i+ threadIdx.x] = currentChild;
					if ((currentChild >= n) && (tempMass[i+ threadIdx.x] = mass[currentChild]) < 0.0f)
					{
						break;
					}
				}

				if (i ==8) //been through all 8 child
				{
					
					px = 0.0f;
					py = 0.0f;
					pz = 0.0f;

					cm = 0.0f;
					counter = 0;

					for (i = 0; i < 8; i++)
					{
						currentChild = tempChild[i + threadIdx.x];
						if (currentChild >=0)
						{
							if (currentChild>= n)
							{
								m = tempMass[i + threadIdx.x];
								counter += count[currentChild];

							}
							else
							{
								m = mass[currentChild];
								counter++;
							}

							//now adding child's data

							cm += m;
							px += pArray->position.x;
							py += pArray->position.y;
							pz += pArray->position.z;
							__threadfence();
							mass[index] = cm;
						}
					}
					index += stride;
				}
				index = restart;
			}

			success = false;
			j = 0;

			while (index <= numNodes)
			{
				if (mass[index] >= 0.0f)
				{
					index += stride;
				}
				else
				{
					if (j == 0)
					{
						j = 8;
						for (int i = 0; i < 8; i++)
						{
							currentChild = child[index * 8 + i];
							tempChild[i + threadIdx.x] = currentChild;
							if ((currentChild < n) || ((tempMass[i + threadIdx.x] = mass[currentChild]) >= 0.0f))
							{
								j--;
							}
						}

					}
					else
					{
						j = 8;
						for (int i = 0; i < 8; i++)
						{
							currentChild = tempChild[i + threadIdx.x];
							if ((currentChild < n) || (tempMass[i  + threadIdx.x] >= 0.0f) || ((tempMass[i  + threadIdx.x] = mass[currentChild]) >= 0.0f))
							{
								j--;
							}
						}
					}

					if (j == 0)
					{
						px = 0.0f;
						py = 0.0f;
						pz = 0.0f;

						cm = 0.0f;
						counter = 0;

						for (int i = 0; i < 8; i++)
						{
							currentChild = tempChild[i + threadIdx.x];
							if (currentChild >= 0)
							{
								if (currentChild >= n)
								{
									m = tempMass[i + threadIdx.x];
									counter += count[currentChild];
								}
								else
								{
									m = mass[currentChild];
									counter++;
								}

								//now child's 
								cm += m;
								px += pArray->position.x;
								py += pArray->position.y;
								pz += pArray->position.z;
							}
						}
						count[index] = counter;
						m = 1.0f/ cm;
						pArray[index].position.x = px * m;
						pArray[index].position.y = py * m;
						pArray[index].position.z = pz * m;
						success = true;
					}
				}
				__syncthreads();  // __threadfence();
				if (success)
				{
					mass[index] = cm;
					index += stride;
					success = true;
				}
			}
		}
	}

}


__global__ void CalculateForces(int* child, int* count, float* mass, float* cmx, float* cmy, float* cmz, int numNodes, int n, Particle* pArray,float theta, float timeStep, float rootSideLegnth)
{
	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;  //index of body based on thread 
	int stride = blockDim.x * gridDim.x; //stride legnth based on number of threads
	bool success;
	int current;
	float sideLegnth;
	int depth;

	float mult;
	float ax;
	float ay;
	float az;

	float g = 6.67408e-11f; //grav constant
	
	int stack[MAX_N];
	int depthStack[MAX_N];
	int top;
	while (bodyIndex < n) //loop all bodies in thread
	{
		//starting at root's children
		top = -1;
		depth = 1;
		sideLegnth = rootSideLegnth * 0.5f;


		for (int i = 0; i < 8; i++)
		{
			top++;
			stack[top] = numNodes* 8 + i; //push root's children
			depthStack[top] = 1;
		}


		while (top >= 0) //while stack is not empty
		{
			current = stack[top];
			depth = depthStack[top];
			if (count[current] == 0 )
			{
				//empty node so pop
				top--;
			}
			else
			{
				float diffX = pArray[bodyIndex].position.x - cmx[current];
				float diffY = pArray[bodyIndex].position.y - cmz[current];
				float diffZ = pArray[bodyIndex].position.z - cmz[current];
				float distance = sqrtf(diffX * diffX + diffY * diffY + diffZ * diffZ);

				if (count[current] ==1) //leaf node so calc forces
				{
					//calculate force
					mult = (g * mass[current]) / (distance * distance * distance);
					ax  =(diffX*mult*timeStep);
					ay  =(diffX*mult*timeStep);
					az  = (diffX*mult*timeStep);


					pArray[bodyIndex].velocity.x -= ax;
					pArray[bodyIndex].velocity.y -= ay;
					pArray[bodyIndex].velocity.z -= az;


					top--;//pop since now calculated

				}
				else 
				{
					sideLegnth = rootSideLegnth * powf(0.5, depth); //sidelegnth based on depth

					if (sideLegnth /distance < theta)
					{
						//calc forces
											//calculate force
						mult = (g * mass[current]) / (distance * distance * distance);
						ax = (diffX * mult * timeStep);
						ay = (diffX * mult * timeStep);
						az = (diffX * mult * timeStep);
						pArray[bodyIndex].velocity.x -= ax;
						pArray[bodyIndex].velocity.y -= ay;
						pArray[bodyIndex].velocity.z -= az;
						//now pop
						top--;
					}
					else
					{
						top--;//pop the parent node
						//need to add this node's children to be traversed
						for (int i = 0; i < 8; i++)
						{
							top++;
							stack[top] = child[current] * 8 + i; //push children
							depthStack[top] = depth + 1;//depth is next depth;
						}
					}
				}

			}
		}
		bodyIndex += stride;
	}

}


__global__ void IntegrateBH(unsigned int n, Particle* pArray, float timeStep)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x; //id of current thread
	int stride = blockDim.x * gridDim.x; //used to stride by number of threads

	float g = 6.67408e-11f; //grav constant

	for (int i = id; i < n; i += stride) //this will loop in i, incrementing by number of threads in parallel
	{
		
		float vdt[3] = { pArray[i].velocity.x * timeStep,pArray[i].velocity.y * timeStep,pArray[i].velocity.z * timeStep };

		pArray[i].position.x = pArray[i].position.x + vdt[0];
		pArray[i].position.y = pArray[i].position.y + vdt[1];
		pArray[i].position.z = pArray[i].position.z + vdt[2];
	}
}
