#include "BH.cuh"

#include "implementKernel.cuh"




void GPUCalls::InitDevice(int n)
{
	threadsPerBlock = 256;
	numberOfBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
}


void GPUCalls::DoFoo(int n , Particle* particles)
{
	



	AllPairs << <threadsPerBlock, numberOfBlocks >> > (n, particles, 0.1f);

	cudaDeviceSynchronize();



}

