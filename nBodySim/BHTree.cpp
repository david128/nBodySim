#include "BHTree.h"

BHTree::BHTree(float halfSide)
{
	root = new Node();
	root->position = Vector3(halfSide, halfSide, halfSide);
	root->sideLegnth = halfSide * 2.0f;;
	
}

BHTree::~BHTree()
{
}

void BHTree::ConstructTree (std::vector<Particle*>* particles)
{
	root->particles = particles;
	root->particleCount = particles->size();
	if (particles->size() > 1)
	{
		SplitNode(root);
	}
	
	
}

void BHTree::SplitNode(Node* currentNode)
{
	Node* children[8];	
	float halfSide = currentNode->sideLegnth * 0.5;
	Vector3 parentCentre = Vector3(currentNode->position.x - halfSide, currentNode->position.y - halfSide, currentNode->position.z - halfSide); //pos is centre +half side in pos; centre = pos -halfside
	if (parentCentre.x == 0.0f || parentCentre.y == 0.0f || parentCentre.z == 0.0f)
	{
		parentCentre += Vector3(0.01f, 0.01f, 0.01f); //alter to avoid dividing by 0
	}
	//create 8 nodes
	for (int i = 0; i < 8; i++)
	{
		children[i] = new Node();
		children[i]->particles = new std::vector<Particle*>();
		children[i]->sideLegnth = halfSide;
		children[i]->FindLocalPosition(i);
		
	}
	
	
	//assign all particles to appropriate node
	for (int i = 0; i < currentNode->particleCount; i++)
	{
		//pos -centre point to find if coordinates are - or + directions from centre
		Vector3 dir = currentNode->particles->at(i)->position - parentCentre;

		dir = Vector3((dir.x / abs(dir.x)), (dir.y / abs(dir.y)), (dir.z / abs(dir.z)));
		bool placed = false;
		int j = 0;
		while (!placed)
		{
			if (dir.equals(children[j]->localPosition))
			{
				children[j]->particles->push_back(currentNode->particles->at(i));
				children[j]->particleCount++;
				placed = true;
			}
			j++;
		}

	}

	//find avg mass and position for all nodes and recursively split if required
	for (int i = 0; i < 8; i++)
	{
		if (children[i]->particleCount == 0)
		{
			children[i]->particle = nullptr;
			
		}
		else if (children[i]->particleCount == 1)
		{
			children[i]->particle = children[i]->particles->at(0);
			children[i]->averageMass = children[i]->particle->mass;
			children[i]->averagePos = children[i]->particle->position;
		}
		else
		{
			//sum masses and positions
			for (int j = 0; j < children[i]->particleCount; j++)
			{
				children[i]->averageMass += children[i]->particles->at(j)->mass;
				children[i]->averagePos += children[i]->particles->at(j)->position;
			}
			//find average
			children[i]->averageMass = children[i]->averageMass / children[i]->particleCount;
			children[i]->averagePos.scale(1.0f/ (float)children[i]->particleCount);


			SplitNode(children[i]); //further split node until 1 or 0 particles
		}
		
	}

}

void Node::FindLocalPosition(int i)
{
	switch (i)
	{
	case 0:
		localPosition = Vector3(1, 1, 1);
		position = Vector3(sideLegnth, sideLegnth, sideLegnth);

		break;
	case 1:
		localPosition = Vector3(1, -1, -1);
		position = Vector3(sideLegnth, 0.0f, 0.0f);
		break;
	case 2:
		localPosition = Vector3(1, -1, 1);
		position = Vector3(sideLegnth, 0.0f, sideLegnth);
		break;
	case 3:
		localPosition = Vector3(1, 1, -1);
		position = Vector3(sideLegnth, sideLegnth, 0.0f);
		break;
	case 4:
		localPosition = Vector3(-1, 1, 1);
		position = Vector3(0.0f, sideLegnth, sideLegnth);
		break;
	case 5:
		localPosition = Vector3(-1, 1,- 1);
		position = Vector3(0.0f, sideLegnth, 0.0f);
		break;
	case 6:
		localPosition = Vector3(-1, -1, 1);
		position = Vector3(0.0f, 0.0f, sideLegnth);
		break;
	case 7:
		localPosition = Vector3(-1, -1,- 1);
		position = Vector3(0.0f, 0.0f, 0.0f);
		break;
	}
}
