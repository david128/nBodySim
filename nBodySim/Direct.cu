﻿#include "Direct.cuh"
#include "implementKernel.cuh"





void DirectGPU::InitDevice(int n)
{
	threadsPerBlock = 256;
	numberOfBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
}


void DirectGPU::AllPairsEuler(int n , Particle* particles, float timeStep)
{
	



	AllPairs << <threadsPerBlock, numberOfBlocks >> > (n, particles, timeStep);

	cudaDeviceSynchronize();



}

