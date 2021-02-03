#pragma once
#include <vector>
#include "Vector3.h"
#include "Particle.h"

struct Node {
    int particleCount;
    Node* children[8];
    Vector3 averagePos;
    float averageMass;
    float sideLegnth;
    Vector3 position;
    Vector3 localPosition;
    Particle* particle;
    std::vector<Particle*>* particles;

    void FindLocalPosition(int i);
};

class BHTree
{
public:

    BHTree(float side);
    ~BHTree();

    void ConstructTree(std::vector<Particle*>* particles);
    void SplitNode(Node* currentNode);
private:

    Node* root;
    Vector3 extents;
};


