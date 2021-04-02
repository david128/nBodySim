#pragma once
#include "Vector3.h"
#include "Particle.h"

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
      
};

class BarnesHutGPU
{
public:

    void SetExtents(float extents);
    void ConstructTree();
    void InseertParticle();


private:
	int threadsPerBlock;
	int numberOfBlocks;

}

