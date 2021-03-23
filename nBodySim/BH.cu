#include "BH.cuh"

#include "implementKernel.cuh"







void BHParallelTree::DoFoo(int n , Particle* particles)
{
	


	//int bytes = n * sizeof(Particle);
	//Particle* pArray;
	//
	//
	//cudaMallocManaged(&pArray, bytes);

	//for (int i = 0; i < n; i++)
	//{
	//	pArray[i] = *particles->at(i);
	//}


	//int* f;

	//cudaMalloc(&f, 2 * sizeof(int));

	//printf("hello Dofoo");
	AllPairs << <1, 1 >> > (n, particles, 0.1f);

	cudaDeviceSynchronize();

	//float y = pArray[1].acceleration.y;

	//cudaFree(pArray);

}

