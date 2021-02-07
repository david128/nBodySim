#pragma once
#include <vector>
#include "Vector3.h"
#include "Particle.h"

struct Node {


    int particleCount;
    //Node* children[8];
    std::vector<Node*> children;
    Vector3 averagePos;
    float averageMass;
    float sideLegnth;
    Vector3 position;
    Vector3 localPosition;
    Particle* particle;
    std::vector<Particle*> particles;

    void FindLocalPosition(int i, Vector3 parentCentre);
};

class BHTree
{
public:

    BHTree(float side);
    ~BHTree();

    void ConstructTree(std::vector<Particle*>* particles);
    void SplitNode(Node* currentNode);
    void DeleteNode(Node* currentNode);
    void DeleteTree();
private:

    Node root;
    Vector3 extents;
};


