#pragma once
#include <vector>
#include "Vector3.h"

struct Node {
    int particleCount;
    std::vector<Node*> children;
    Vector3 averagePos;
    float averageMass;
    float sideLegnth;
    Vector3 position;
};

class BHTree
{
public:

    BHTree();
    ~BHTree();

    void ConstructTree();

private:

    Node* root;

};


