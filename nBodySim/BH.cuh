
#pragma once

#include "BHTree.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"


class BHParallelTree : BHTree
{
public:
	void ConstructTree(std::vector<Particle*>* particles);
	void SplitOnce();
	void SplitNodeInP();
	void DoFoo();

};


