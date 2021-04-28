#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "Particle.h"


__global__ void EulerAcceleration(unsigned int n, Particle* pArray, float timeStep);

__global__ void EulerPosition(unsigned int n, Particle* pArray, float timeStep);

