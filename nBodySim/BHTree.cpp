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

	//create 8 nodes
	for (int i = 0; i < 8; i++)
	{
		children[i] = new Node();
		children[i]->FindLocalPosition(i);
		children[i]->sideLegnth = halfSide;
		Vector3 translation = children[i]->localPosition;
		translation.scale(halfSide);//find translation from centre that will give top right back coordinate
		children[i]->position = parentCentre + translation;
		
	}
	
	//assign all particles to appropriate node
	for (auto particle : *root->particles)
	{
		//pos -centre point to find if coordinates are - or + directions from centre
		Vector3 dir = particle->position - parentCentre;
		dir = (dir.x / abs(dir.x), dir.y / abs(dir.y), dir.z / abs(dir.z));
		bool placed = false;
		int i = 0;
		while (!placed)
		{
			if (dir.equals(children[i]->localPosition))
			{
				children[i]->particles->push_back(particle);
				children[i]->particleCount++;
				placed = true;
			}
			i++;
		}

	}

	//find avg mass and position for all nodes and recursively split if required
	for (int i = 0; i < 8; i++)
	{
		if (children[i]->particleCount == 0)
		{
			children[i]->particle = nullptr;
			
		}
		else if (children[i]->particleCount == 0)
		{
			children[i]->particle = children[i]->particles->at(0);
			children[i]->averageMass = children[i]->particle->mass;
			children[i]->averagePos = children[i]->particle->position;
		}
		else
		{
			//sum masses and positions
			for (int i = 0; i < children[i]->particleCount; i++)
			{
				children[i]->averageMass += children[i]->particles->at(i)->mass;
				children[i]->averagePos += children[i]->particles->at(i)->position;
			}
			//find average
			children[i]->averageMass = children[i]->averageMass / children[i]->particleCount;
			children[i]->averagePos.scale(1.0f/ children[i]->particleCount);


			SplitNode(children[i]);
		}
		
	}

}

void Node::FindLocalPosition(int i)
{
	switch (i)
	{
	case 0:
		localPosition = Vector3(1, 1, 1);
		break;
	case 1:
		localPosition = Vector3(1, -1, -1);
		break;
	case 2:
		localPosition = Vector3(1, -1, 1);
		break;
	case 3:
		localPosition = Vector3(1, 1, -1);
		break;
	case 4:
		localPosition = Vector3(-1, 1, 1);
		break;
	case 5:
		localPosition = Vector3(-1, 1,- 1);
		break;
	case 6:
		localPosition = Vector3(-1, -1, 1);
		break;
	case 7:
		localPosition = Vector3(-1, -1,- 1);
		break;
	}
}
