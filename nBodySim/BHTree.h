#pragma once
#include <vector>
#include "Vector3.h"
#include "Particle.h"
#include "Solver.h"

//node structure for tree
struct Node {
    int particleCount;
    std::vector<Node*> children;
    Vector3 averagePos;
    float combinedMass;
    float sideLegnth;
    Vector3 position;
    std::vector<Particle*> particles;
};

class BHTree: 
    public Solver
{
public:

    BHTree(float gravConst, float th, Particle* particles, int n);
    ~BHTree();
    
    void ConstructTree(Particle* particles, int n);
    void SplitNode(Node* currentNode);
    void DeleteNode(Node* currentNode);
    void DeleteTree();
    void CalculateForces(Particle* particles, int n, float timeStep);
    void TraversNode(Particle* particle, float theta, Node* node, float timeStep);
    void UpdatePositions(Particle* particles, float timeStep, int n);

    void Solve(Particle* particles, float timeStep, int n);

protected:

    float theta;
    Node root;
    float maxPos;
};


