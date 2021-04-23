#pragma once

#include "Particle.h"

#include "NodeGPU.h"


class BarnesHutGPU
{
public:

	void InitRoot(int n,float halfSide);
	void SetExtents(float extents);
	void ConstructTree(int n, Particle* pArray);
	void InseertParticle();


private:
	int threadsPerBlock;
	int numberOfBlocks;

	int numNodes;


	int* children;
	float* masses;
	int* counter;
	float* cmx;
	float* cmy;
	float* cmz;

	NodeGPU* root;


};


