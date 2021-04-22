#pragma once
#include "Particle.h"
#include "Vector3.h"

struct NodeGPU {


	int particleCount;
	NodeGPU* children[8];
	Vector3 averagePos;
	float combinedMass;
	float sideLegnth;
	Vector3 position;
	Vector3 localPosition;
	Particle* particle;
	Particle* particles;
	bool locked;
	int index;
};


struct Level
{
	int treeLevel;
	int minIndex;
	int maxIndex;
	int* indexes;
};

