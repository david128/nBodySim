#include "BHTree.h"


BHTree::BHTree(float gravConst,float th, Particle* particles, int n)
{
	//set maxPos to initial 0
	maxPos = 0;

	
	for (int i = 0; i < n; i++)
	{
		root.particles.push_back(&particles[i]);//insert this particle to root
		//find max extent of system
		if (abs(particles[i].position.x) > maxPos) { maxPos = abs(particles[i].position.x);}
		if (abs(particles[i].position.y) > maxPos) { maxPos = abs(particles[i].position.y);}
		if (abs(particles[i].position.z) > maxPos) { maxPos = abs(particles[i].position.z);}
	}

	//set g and theta
	root.particleCount = n;
	g = gravConst;
	theta = th;

}

BHTree::~BHTree()
{
}

void BHTree::ConstructTree (Particle* particles, int n)
{

	//set root pos to centre of system
	root.position = Vector3(0.0f, 0.0f, 0.0f);
	root.sideLegnth = maxPos * 2.0f;

	if (n > 1)
	{
		//start by splitting root node
		SplitNode(&root);
	}
	
}


void BHTree::DeleteTree()
{
	//if root has children start delete process at root
	if (root.children.size() != 0)
	{
		DeleteNode(&root);
	}
	
}

void BHTree::CalculateForces(Particle* particles, int n, float timeStep)
{

	for (int i = 0; i < n; i++) //for all particles find forces applied
	{
		TraversNode(&particles[i], theta,&root, timeStep);//start at root
	}

}

void BHTree::TraversNode(Particle* particle, float theta, Node* currentNode, float timeStep)
{
	Vector3 vdt;
	for (auto node : currentNode->children)
	{
		if (node != NULL)//if empty node then we can skip as not particles to calculate force of
		{
			if (node->particleCount == 1) //external node we can use nodes avgs because these are values from the one single particle
			{
				if (node->particles[0] != particle) //do not calculate force of particle on self
				{
					vdt =CalculateAcceleration(particle->position, node->averagePos, node->combinedMass);
					vdt.scale(timeStep);
					particle->velocity += vdt;
				}
				
			}
			else
			{
				Vector3 diff = particle->position - node->averagePos;
				float dist = diff.length();//get distance between points
				if ((node->sideLegnth / dist) < theta)
				{
				
					vdt = CalculateAcceleration(particle->position, node->averagePos, node->combinedMass);
					vdt.scale(timeStep);
					particle->velocity += vdt;
				}
				else
				{
					//too close, so need to traverse node further
					TraversNode(particle, theta, node, timeStep);
				}

			}
		}
	}
}

void BHTree::UpdatePositions(Particle* particles, float timeStep, int n)
{
	for (int i = 0; i < n; i++)
	{
		Vector3 vDt = particles[i].velocity;
		vDt.scale(timeStep);
		particles[i].position += vDt;
		//expand if needed
		if (abs(particles[i].position.x) > maxPos) { maxPos = abs(particles[i].position.x); } 
		if (abs(particles[i].position.y) > maxPos) { maxPos = abs(particles[i].position.y); }
		if (abs(particles[i].position.z) > maxPos) { maxPos = abs(particles[i].position.z); }
	}
	
}



void BHTree::Solve(Particle* particles, float timeStep, int n)
{
	DeleteTree(); //delete previous tree
	ConstructTree(particles, n); //construct tree based on particles
	CalculateForces(particles,n, timeStep); //calculate forces, approximating based on tree
	UpdatePositions(particles, timeStep, n); //update positions using euler

}

void BHTree::SplitNode(Node* currentNode)
{
	
	float halfSide = currentNode->sideLegnth * 0.5f; //find half side 
	
	//create 8 nodes
	for (int i = 0; i < 8; i++)
	{
		currentNode->children.push_back(new Node());
		currentNode->children[i]->sideLegnth = halfSide; //set l to 05* parents side
		
	}
	float quarterSide = halfSide * 0.5f;
	//set positions
	currentNode->children[0]->position = currentNode->position + Vector3(-quarterSide, -quarterSide, -quarterSide);
	currentNode->children[1]->position = currentNode->position + Vector3(quarterSide, -quarterSide, -quarterSide);
	currentNode->children[2]->position = currentNode->position + Vector3(-quarterSide, quarterSide, -quarterSide);
	currentNode->children[3]->position = currentNode->position + Vector3(quarterSide, quarterSide, -quarterSide);
	currentNode->children[4]->position = currentNode->position + Vector3(-quarterSide, -quarterSide, quarterSide);
	currentNode->children[5]->position = currentNode->position + Vector3(quarterSide, -quarterSide, quarterSide);
	currentNode->children[6]->position = currentNode->position + Vector3(-quarterSide, quarterSide, quarterSide);
	currentNode->children[7]->position = currentNode->position + Vector3(quarterSide, quarterSide, quarterSide);
	
	int childIndex;
	//assign all particles to appropriate node
	for (int i = 0; i < currentNode->particleCount; i++)
	{
					
		//find child index
		childIndex = 0;
		if (currentNode->position.x < currentNode->particles.at(i)->position.x) { childIndex = 1; } 
		if (currentNode->position.y < currentNode->particles.at(i)->position.y) { childIndex |= 2; }
		if (currentNode->position.z < currentNode->particles.at(i)->position.z) { childIndex |= 4; }

		//insert this particle
		currentNode->children[childIndex]->particleCount++;
		currentNode->children[childIndex]->particles.push_back(currentNode->particles[i]);

	}

	//find avg mass and position for all nodes and recursively split if required
	for (int i = 0; i < 8; i++)
	{

			if (currentNode->children[i]->particleCount == 0)
			{
				delete currentNode->children[i]; //delete, do not need to store
				currentNode->children[i] = NULL; //node is empty

			}
			else if (currentNode->children[i]->particleCount == 1)
			{
				//only one node
				currentNode->children[i]->combinedMass = currentNode->children[i]->particles.at(0)->mass; 
				currentNode->children[i]->averagePos = currentNode->children[i]->particles.at(0)->position;
			}
			else
			{
				//sum masses and weighted positions
				for (int j = 0; j < currentNode->children[i]->particleCount; j++)
				{
					currentNode->children[i]->combinedMass += currentNode->children[i]->particles.at(j)->mass;
					Vector3 weightedPosition = currentNode->children[i]->particles.at(j)->position;
					weightedPosition.scale(currentNode->children[i]->particles.at(j)->mass);
					currentNode->children[i]->averagePos += weightedPosition; //add position*mass
				}
				//find weighted average
				currentNode->children[i]->averagePos.scale(1.0f / currentNode->children[i]->combinedMass); 

				SplitNode(currentNode->children[i]); //further split node until 1 or 0 particles in children
				
			}
	}
	
}


void BHTree::DeleteNode(Node* currentNode)
{

	for (int i = 0; i < 8; i++) //loop through children of cur node
	{
		if (currentNode->children[i] != NULL) //nothing to delete
		{
			if (currentNode->children[i]->particleCount > 1)
			{
				DeleteNode(currentNode->children[i]); //recursively delete node's children
			}
			
			delete currentNode->children[i]; //delete this child
			currentNode->children[i] = NULL; //set pntr to NULL
		}

	}
	currentNode->children.clear();
	
	
}

