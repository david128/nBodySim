
#pragma once
#include "Particle.h"
#include "Solver.h"



class DirectGPU :
	public Solver
{
public:
	DirectGPU(int n);
	void Solve(Particle* particles, float timeStep, int n);
	void InitDevice(int n);	

private:
	int threadsPerBlock;
	int numberOfBlocks;
};


