#include "BarnesHut.cuh"
#include "implementKernel.cuh"

void BarnesHutGPU::InitRoot(int n,float halfSide)
{

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,0);

	numNodes = n * 2;

	numberOfBlocks = deviceProp.multiProcessorCount;

	if (numNodes < 1024 * numberOfBlocks) numNodes = 1024 * numberOfBlocks;
	/*while ((numNodes & (WARPSIZE - 1)) != 0) nnodes++;
	nnodes--;*/
	int bytes = sizeof(NodeGPU);

	cudaMallocManaged(&root, bytes);


	//cudaMalloc((void**)&children, 8 * (numNodes + 1) * sizeof(int));
	bool worked = true;
	if (cudaSuccess != cudaMalloc((void**)&children, sizeof(int) * (numNodes + 1) * 8))
	{
		worked = false;
	}



	root->position = Vector3(halfSide, halfSide, halfSide);
	root->sideLegnth = halfSide * 2.0f;

}


void BarnesHutGPU::ConstructTree(int n, Particle* pArray)
{
	Level startLevel;
	startLevel.maxIndex = 0;
	startLevel.minIndex = 0;
	startLevel.treeLevel = 0;
	


	
	rootKernel << <1, 1 >> > (children, numNodes);
	cudaDeviceSynchronize();
	clearKernel << <1, 1 >> > (children, numNodes,n);
	cudaDeviceSynchronize();
	buildTreeInsertion << < 1,1 >> > (root, n, pArray, children,numNodes);
	cudaDeviceSynchronize();

	int f = 0;
	f++;

	//cudaFree(currentNode);
}