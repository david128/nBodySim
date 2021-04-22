#pragma once

class NodeGPU {

public:
	int particleCount;
	NodeGPU* children[8];
	float averagePos[3];
	float combinedMass;
	float sideLegnth;
	float position[3];
	float localPosition[3];
	Particle* particle;
	Particle* particles;
	bool locked;
};


class Level
{
public:
	int treeLevel;
	int minIndex;
	int maxIndex;
};