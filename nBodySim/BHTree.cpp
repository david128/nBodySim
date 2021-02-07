#include "BHTree.h"

BHTree::BHTree(float halfSide)
{
	
	root.position = Vector3(halfSide, halfSide, halfSide);
	root.sideLegnth = halfSide * 2.0f;;
	
}

BHTree::~BHTree()
{
}

void BHTree::ConstructTree (std::vector<Particle*>* particles)
{

	
	root.particles = *particles;
	root.particleCount = particles->size();
	if (particles->size() > 1)
	{
		SplitNode(&root);
	}
	
	
}

void BHTree::DeleteTree()
{
	DeleteNode(&root);
}

void BHTree::SplitNode(Node* currentNode)
{
	
	float halfSide = currentNode->sideLegnth * 0.5;
	Vector3 parentCentre = Vector3(currentNode->position.x - halfSide, currentNode->position.y - halfSide, currentNode->position.z - halfSide); //pos is centre +half side in pos; centre = pos -halfside
	if (parentCentre.x == 0.0f || parentCentre.y == 0.0f || parentCentre.z == 0.0f)
	{
		parentCentre += Vector3(0.01f, 0.01f, 0.01f); //alter to avoid dividing by 0
	}
	//create 8 nodes
	for (int i = 0; i < 8; i++)
	{
		currentNode->children.push_back(new Node());
		currentNode->children[i]->sideLegnth = halfSide;
		currentNode->children[i]->FindLocalPosition(i, parentCentre);
		
	}
	
	
	//assign all particles to appropriate node
	for (int i = 0; i < currentNode->particleCount; i++)
	{
		//pos -centre point to find if coordinates are - or + directions from centre
		Vector3 dir = currentNode->particles[i]->position - parentCentre;


		dir = Vector3((dir.x / abs(dir.x)), (dir.y / abs(dir.y)), (dir.z / abs(dir.z)));
		bool placed = false;
		int j = 0;
		while (!placed)
		{
			if (dir.equals(currentNode->children[j]->localPosition))
			{
				currentNode->children[j]->particles.push_back(currentNode->particles[i]);
				currentNode->children[j]->particleCount++;
				placed = true;
			}
			j++;
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
			currentNode->children[i]->particle = currentNode->children[i]->particles[0];
			currentNode->children[i]->averageMass = currentNode->children[i]->particle->mass;
			currentNode->children[i]->averagePos = currentNode->children[i]->particle->position;
		}
		else
		{
			//sum masses and positions
			for (int j = 0; j < currentNode->children[i]->particleCount; j++)
			{
				currentNode->children[i]->averageMass += currentNode->children[i]->particles[j]->mass;
				currentNode->children[i]->averagePos += currentNode->children[i]->particles[j]->position;
			}
			//find average
			currentNode->children[i]->averageMass = currentNode->children[i]->averageMass / currentNode->children[i]->particleCount;
			currentNode->children[i]->averagePos.scale(1.0f/ (float)currentNode->children[i]->particleCount);


			SplitNode(currentNode->children[i]); //further split node until 1 or 0 particles
		}
		
	}
	
}

void BHTree::DeleteNode(Node* currentNode)
{

	for (int i = 0; i < 8; i++) //loop through children of cur node
	{
		if (currentNode->children[i]->particleCount > 1) //if have more than 1 children then recursively delete
		{
			DeleteNode(currentNode->children[i]);
		}
		delete currentNode->children[i]; //delete this child
		currentNode->children[i] = NULL; //delete this child
		
	}
	currentNode->children.clear();
	
	
}

void Node::FindLocalPosition(int i, Vector3 parentCentre)
{
	switch (i)
	{
	case 0:
		localPosition = Vector3(1, 1, 1);
		position = parentCentre + (sideLegnth, sideLegnth, sideLegnth);

		break;
	case 1:
		localPosition = Vector3(1, -1, -1);
		position = parentCentre + (sideLegnth, 0.0f, 0.0f);
		break;
	case 2:
		localPosition = Vector3(1, -1, 1);
		position = parentCentre + (sideLegnth, 0.0f, sideLegnth);
		break;
	case 3:
		localPosition = Vector3(1, 1, -1);
		position = parentCentre + (sideLegnth, sideLegnth, 0.0f);
		break;
	case 4:
		localPosition = Vector3(-1, 1, 1);
		position = parentCentre + (0.0f, sideLegnth, sideLegnth);
		break;
	case 5:
		localPosition = Vector3(-1, 1,- 1);
		position = parentCentre + (0.0f, sideLegnth, 0.0f);
		break;
	case 6:
		localPosition = Vector3(-1, -1, 1);
		position = parentCentre + (0.0f, 0.0f, sideLegnth);
		break;
	case 7:
		localPosition = Vector3(-1, -1,- 1);
		position = parentCentre + (0.0f, 0.0f, 0.0f);
		break;
	}
}
