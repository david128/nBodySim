#pragma once
#include <vector>
#include "Vector3.h"
#include "Particle.h"
#include "Solver.h"


struct Node {


    int particleCount;
    //Node* children[8];
    std::vector<Node*> children;
    Vector3 averagePos;
    float combinedMass;
    float sideLegnth;
    Vector3 position;
    Vector3 localPosition;
    std::vector<Particle*> particles;
    Particle* particle;
   
    void FindLocalPosition(int i, Vector3 parentCentre);
};

class BHTree: 
    public Solver
{
public:

    BHTree(float side, float gravConst,float th);
    ~BHTree();


    void ConstructTree(Particle* particles, int n);
    void SplitNode(Node* currentNode);
    void DeleteNode(Node* currentNode);
    void DeleteTree();
    void CalculateForces(Particle* particles, int n, float timeStep);
    void TraversNode(Particle* particle, float theta, Node* node, float timeStep);
    void CalculateForce(Particle* particle, Vector3 acm, float am, float timeStep);

    void Solve(Particle* particles, float timeStep, int n);

protected:

    float theta;
    Node root;
    Vector3 extents;
    float maxPos;
};


