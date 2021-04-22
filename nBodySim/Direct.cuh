
#pragma once
#include "Particle.h"




class DirectGPU 
{
public:

	void InitDevice(int n);	
	void AllPairsEuler(int n, Particle* particles, float timeStep);


private:
	int threadsPerBlock;
	int numberOfBlocks;
};


