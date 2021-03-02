#pragma once
#include <vector>
#include "Vector3.h"
#include "Particle.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

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
    void ConstructTreeInP(std::vector<Particle*>* particles);
    void SplitNode(Node* currentNode);
    __global__ void SplitNodeInP(Node* currentNode);
    void SplitOnce();
    void DeleteNode(Node* currentNode);
    void DeleteTree();
    void CalculateForces(float theta, std::vector<Particle*>* particles, float timeStep);
    void TraversNode(Particle* particle, float theta, Node* node, float timeStep);
    void CalculateForce(Particle* particle, Vector3 acm, float am, float timeStep);
    void DrawDebug();
    void DrawLines(Node* node);

private:

    Node root;
    Vector3 extents;
    float g;
    float time;
    float maxPos;
};


