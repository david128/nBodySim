#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "Particle.h"
#include "NodeGPU.h"





__global__
void AllPairs(unsigned int n, Particle* pArray, float timeStep);

//__global__ void doFoo();


//__global__
//void buildTreeSplit(Level prevLevel, NodeGPU* nodes);

__global__ void rootKernel(int* child, int numNodes);

__global__ void clearKernel(int* child, int numNodes, int n);

__global__ 
void buildTreeInsertion(NodeGPU* root, int n, Particle* pArray, int* child, int numNodes);
