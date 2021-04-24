#include "BHTree.h"


BHTree::BHTree(float gravConst,float th, Particle* particles, int n)
{
	maxPos = 0;

	for (int i = 0; i < n; i++)
	{
		root.particles.push_back(&particles[i]);
		if (abs(particles[i].position.x) > maxPos) { maxPos = abs(particles[i].position.x);}
		if (abs(particles[i].position.y) > maxPos) { maxPos = abs(particles[i].position.y);}
		if (abs(particles[i].position.z) > maxPos) { maxPos = abs(particles[i].position.z);}
	}

	root.particleCount = n;
	g = gravConst;
	theta = th;

}

BHTree::~BHTree()
{
}



void BHTree::ConstructTree (Particle* particles, int n)
{


	root.position = Vector3(maxPos, maxPos, maxPos);
	root.sideLegnth = maxPos * 2.0f;
	

	if (n > 1)
	{
		SplitNode(&root);
	}
	
}


void BHTree::DeleteTree()
{
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
	DeleteTree();
	ConstructTree(particles, n);
	CalculateForces(particles,n, timeStep);
	UpdatePositions(particles, timeStep, n);

}

void BHTree::SplitNode(Node* currentNode)
{
	
	float halfSide = currentNode->sideLegnth * 0.5;
	Vector3 parentCentre = Vector3(currentNode->position.x - halfSide, currentNode->position.y - halfSide, currentNode->position.z - halfSide); //pos is centre +half side in pos; centre = pos -halfside

	//create 8 nodes
	for (int i = 0; i < 8; i++)
	{
		currentNode->children.push_back(new Node());
		currentNode->children[i]->sideLegnth = halfSide;
		currentNode->children[i]->FindLocalPosition(i, parentCentre);
		
	}
	

	parentCentre += Vector3(0.01f, 0.01f, 0.01f); //alter to avoid dividing by 0


	
	//assign all particles to appropriate node
	for (int i = 0; i < currentNode->particleCount; i++)
	{
		bool inside = true; // if particle is not inside extents of root then do not recursively split it.
		if (abs(currentNode->particles.at(i)->position.x) > root.sideLegnth*0.5f || abs(currentNode->particles.at(i)->position.y) > root.sideLegnth * 0.5f || abs(currentNode->particles.at(i)->position.z) > root.sideLegnth * 0.5f)
		{
			inside = false; //if not inside extents, set inside to false
		}
		//pos -centre point to find if coordinates are - or + directions from centre
		Vector3 dir = currentNode->particles.at(i)->position - parentCentre;

		if (inside) //only place particle if inside extents(otherwise will not correctly place, and will inifinitely run until out of memory crash.
		{
			
			dir = Vector3((dir.x / abs(dir.x)), (dir.y / abs(dir.y)), (dir.z / abs(dir.z)));
			bool placed = false;
			int j = 0;
			while (!placed)
			{
				if (dir.equals(currentNode->children[j]->localPosition))
				{
					currentNode->children[j]->particleCount++;
					currentNode->children[j]->particles.push_back(currentNode->particles[i]);
					//currentNode->children[j]->particles[currentNode->children[j]->particleCount -1] = &currentNode->particles[i]; //add this particle to child

					placed = true;
				}
				j++;
			}
		}

		

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
				currentNode->children[i]->combinedMass = currentNode->children[i]->particles.at(0)->mass;
				currentNode->children[i]->averagePos = currentNode->children[i]->particles.at(0)->position;
			}
			else
			{
				//sum masses and positions
				for (int j = 0; j < currentNode->children[i]->particleCount; j++)
				{
					currentNode->children[i]->combinedMass += currentNode->children[i]->particles.at(j)->mass;
					currentNode->children[i]->averagePos += currentNode->children[i]->particles.at(j)->position;
				}
				//find average
				currentNode->children[i]->averagePos.scale(1.0f / (float)currentNode->children[i]->particleCount);


				SplitNode(currentNode->children[i]); //further split node until 1 or 0 particles
				
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

void Node::FindLocalPosition(int i, Vector3 parentCentre)
{
	switch (i)
	{
	case 0:
		localPosition = Vector3(1, 1, 1);
		position = parentCentre + Vector3(sideLegnth, sideLegnth, sideLegnth);

		break;
	case 1:
		localPosition = Vector3(1, -1, -1);
		position = parentCentre + Vector3(sideLegnth, 0.0f, 0.0f);
		break;
	case 2:
		localPosition = Vector3(1, -1, 1);
		position = parentCentre + Vector3(sideLegnth, 0.0f, sideLegnth);
		break;
	case 3:
		localPosition = Vector3(1, 1, -1);
		position = parentCentre + Vector3(sideLegnth, sideLegnth, 0.0f);
		break;
	case 4:
		localPosition = Vector3(-1, 1, 1);
		position = parentCentre + Vector3(0.0f, sideLegnth, sideLegnth);
		break;
	case 5:
		localPosition = Vector3(-1, 1,- 1);
		position = parentCentre + Vector3(0.0f, sideLegnth, 0.0f);
		break;
	case 6:
		localPosition = Vector3(-1, -1, 1);
		position = parentCentre + Vector3(0.0f, 0.0f, sideLegnth);
		break;
	case 7:
		localPosition = Vector3(-1, -1,- 1);
		position = parentCentre + Vector3(0.0f, 0.0f, 0.0f);
		break;
	}
}
