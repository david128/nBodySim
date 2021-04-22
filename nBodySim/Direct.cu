#include "Direct.cuh"
#include "implementKernel.cuh"





DirectGPU::DirectGPU(int n)
{
	InitDevice(n);
}

void DirectGPU::Solve(Particle* particles, float timeStep, int n)
{
	AllPairsEuler(particles, timeStep, n);
}

void DirectGPU::InitDevice(int n)
{
	threadsPerBlock = 256;
	numberOfBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
}


void DirectGPU::AllPairsEuler(Particle* particles, float timeStep, int n)
{
	
	//sum accelerations
	EulerAcceleration << <threadsPerBlock, numberOfBlocks >> > (n, particles, timeStep);
	//sync since accelerations must be completed before integrating positions
	cudaDeviceSynchronize();
	//Integrate position using new velocity from acceleration
	EulerPosition << <threadsPerBlock, numberOfBlocks >> > (n, particles, timeStep);
	//sync before continuing update
	cudaDeviceSynchronize();

}

