#pragma once
#include <vector>
#include "Vector3.h"
#include "Particle.h"

struct Node {


    int particleCount;
    //Node* children[8];
    std::vector<Node*> children;
    Vector3 averagePos;
    float combinedMass;
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

    BHTree(float side, float gravConst);
    ~BHTree();

    bool Update(float dt, float timeStep);

    void ConstructTree(std::vector<Particle*>* particles);
    void SplitNode(Node* currentNode);
    void DeleteNode(Node* currentNode);
    void DeleteTree();
    void CalculateForces(float theta, std::vector<Particle*>* particles);
    void TraversNode(Particle* particle, float theta, Node* node);
    void CalculateForce(Particle* particle, Vector3 acm, float am);
private:

    Node root;
    Vector3 extents;
    float g;
    float time;
};


