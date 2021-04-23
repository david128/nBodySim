#include "BarnesHut.cuh"
#include "implementKernel.cuh"

void BarnesHutGPU::InitRoot(int n,float halfSide)
{
	
	threadsPerBlock = 256;
	if (threadsPerBlock > n)
	{
		threadsPerBlock = n;
	}
	//numberOfBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

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
	cudaMalloc((void**)&counter, sizeof(int) * (numNodes + 1) * 8);
	cudaMalloc((void**)&masses, sizeof(float) * (numNodes + 1) * 8);

	cudaMalloc((void**)&cmx, sizeof(float) * (numNodes + 1) * 8);
	cudaMalloc((void**)&cmy, sizeof(float) * (numNodes + 1) * 8);
	cudaMalloc((void**)&cmz, sizeof(float) * (numNodes + 1) * 8);


	root->position = Vector3(halfSide, halfSide, halfSide);
	root->sideLegnth = halfSide * 2.0f;

}


void BarnesHutGPU::ConstructTree(int n, Particle* pArray)
{
	Level startLevel;
	startLevel.maxIndex = 0;
	startLevel.minIndex = 0;
	startLevel.treeLevel = 0;


	
	rootKernel << <1, 1 >> > (children, masses, numNodes);
	cudaDeviceSynchronize();
	clearKernel << <threadsPerBlock, numberOfBlocks >> > (children, masses, numNodes,n);
	cudaDeviceSynchronize();
	
	buildTreeInsertion << <32, 2 >> > (root, n, pArray, children,numNodes);
	
	cudaDeviceSynchronize();

	//CalculateForces << <threadsPerBlock, numberOfBlocks >> > (children,counter,masses,cmx,cmy,cmz,numNodes, n, pArray,0.5f,0.5f, root->sideLegnth);
	//cudaDeviceSynchronize();
	//IntegrateBH << <threadsPerBlock, numberOfBlocks >> > (n, pArray, 0.5f);
	cudaDeviceSynchronize();
	int f = 0;
	f++;

	//cudaFree(currentNode);
}