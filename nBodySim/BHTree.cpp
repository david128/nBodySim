#include "BHTree.h"

BHTree::BHTree(float halfSide , float gravConst)
{
	
	root.position = Vector3(halfSide, halfSide, halfSide);
	maxPos = halfSide;
	root.sideLegnth = halfSide * 2.0f;;

	time = 0.0f;
	g = gravConst;
}

BHTree::~BHTree()
{
}

bool BHTree::Update(float dt, float timeStep)
{
	time += dt;

	if (time >= timeStep)
	{
		time = 0.0f;//reset time
		return true; //return true so we now solve

	}
	return false;
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
	if (root.children.size() != 0)
	{
		DeleteNode(&root);
	}
	
}

void BHTree::CalculateForces(float theta, std::vector<Particle*>* particles, float timeStep)
{

	for (int i = 0; i < particles->size(); i++) //for all particles find forces applied
	{
		TraversNode(particles->at(i), theta,&root, timeStep);//start at root
		particles->at(i)->Update();
	}

}

void BHTree::TraversNode(Particle* particle, float theta, Node* currentNode, float timeStep)
{
	for (auto node : currentNode->children)
	{
		if (node != NULL)//if empty node then we can skip as not particles to calculate force of
		{
			if (node->particleCount == 1) //external node we can use nodes avgs because these are values from the one single particle
			{
				if (node->particle != particle) //do not calculate force of particle on self
				{
					CalculateForce(particle, node->averagePos, node->combinedMass, timeStep);
				}
				
			}
			else
			{
				Vector3 diff = particle->position - node->averagePos;
				float dist = diff.length();//get distance between points
				if ((node->sideLegnth / dist) < theta)
				{

					CalculateForce(particle, node->averagePos, node->combinedMass, timeStep); //suitably far so can use avg mass and CoM
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

void BHTree::CalculateForce(Particle* particle, Vector3 acm, float cm, float timeStep ) //using euler method
{
	Vector3 diff = particle->position - acm;
	float dist = diff.length(); //get distance

	float mult = (g * cm) / (dist * dist * dist); //multiplier  (g * mass )/ (distance ^3)

	Vector3 multDiff = Vector3(mult * diff.getX(), mult * diff.getY(), mult * diff.getZ()); //multiply  vector by multiplier to get force
	multDiff.scale(timeStep);
	particle->velocity = particle->velocity - multDiff; //new V = old v + acceleration due to gravity
	
	
}

void BHTree::DrawDebug()
{

	DrawLines(&root);

}

void BHTree::DrawLines(Node* node)
{


	glBegin(GL_LINES);

	glVertex3f(-10000, 10000, 0);
	glVertex3f(10000, 10000, 0);
	glVertex3f(10000, -10000, 0);
	glVertex3f(-10000, 10000, 0);


	glEnd();

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
	
	if (parentCentre.x == 0.0f || parentCentre.y == 0.0f || parentCentre.z == 0.0f)
	{
		parentCentre += Vector3(0.01f, 0.01f, 0.01f); //alter to avoid dividing by 0
	}

	
	//assign all particles to appropriate node
	for (int i = 0; i < currentNode->particleCount; i++)
	{
		bool inside = true; // if particle is not inside extents of root then do not recursively split it.
		if (abs(currentNode->particles[i]->position.x) > root.sideLegnth*0.5f || abs(currentNode->particles[i]->position.y) > root.sideLegnth * 0.5f || abs(currentNode->particles[i]->position.z) > root.sideLegnth * 0.5f)
		{
			inside = false; //if not inside extents, set inside to false
		}
		//pos -centre point to find if coordinates are - or + directions from centre
		Vector3 dir = currentNode->particles[i]->position - parentCentre;

		if (inside) //only place particle if inside extents(otherwise will not correctly place, and will inifinitely run until out of memory crash.
		{
			
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
				currentNode->children[i]->combinedMass = currentNode->children[i]->particle->mass;
				currentNode->children[i]->averagePos = currentNode->children[i]->particle->position;
			}
			else
			{
				//sum masses and positions
				for (int j = 0; j < currentNode->children[i]->particleCount; j++)
				{
					currentNode->children[i]->combinedMass += currentNode->children[i]->particles[j]->mass;
					currentNode->children[i]->averagePos += currentNode->children[i]->particles[j]->position;
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
