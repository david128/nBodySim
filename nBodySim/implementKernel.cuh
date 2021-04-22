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

__global__ void rootKernel(int* child,float* mass, int numNodes);

__global__ void clearKernel(int* child, float* mass, int numNodes, int n);

__global__ void SumKernel(int* child, int* count, int numNodes, int n, float* mass, Particle* pArray);

__global__ 
void buildTreeInsertion(NodeGPU* root, int n, Particle* pArray, int* child, int numNodes);

__global__ void CalculateForces(int* child, int* count, float* mass, float* cmx, float* cmy, float* cmz, int numNodes, int n, Particle* pArray, float theta, float timeStep, float rootSideLegnth);

__global__ void IntegrateBH(unsigned int n, Particle* pArray, float timeStep);
