
#pragma once
#include "Particle.h"
#include <vector>


class GPUCalls 
{
public:

	void InitDevice(int n);	
	void DoFoo(int n, Particle*  particles);


private:
	int threadsPerBlock;
	int numberOfBlocks;
};


