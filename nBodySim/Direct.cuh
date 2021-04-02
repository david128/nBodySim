
#pragma once
#include "Particle.h"
#include <vector>


class DirectGPU 
{
public:

	void InitDevice(int n);	
	void AllPairsEuler(int n, Particle*  particles);


private:
	int threadsPerBlock;
	int numberOfBlocks;
};


