#include "BH.cuh"
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"


__global__ void SplitNodeInP(Node* currentNode)
{

	//SplitNode(currentNode);
}



void BHParallelTree::ConstructTree(std::vector<Particle*>* particles)
{
	//split node into 8 but do not recursively split further

	int* d_g;
	int maxNodes = particles->size();

	root.particles = *particles;
	root.particleCount = particles->size();
	if (particles->size() > 1)
	{
		SplitOnce();

	}

	
	//in parrallel further split each node
	//cudaMallocManaged((void**)&root,maxNodes * sizeof(Node));
	//cudaMallocManaged(&d_g, sizeof(int));


	
	//SplitNodeInP<<<1, 1, >>>(root.children[1]);
	//addS<<<1, 1, >>>(d_g);
	
	
	//cudaDeviceSynchronize();

	//cudaFree(&root);
	//cudaFree(d_g);

}

void BHParallelTree::SplitOnce()
{
	float halfSide = root.sideLegnth * 0.5;
	Vector3 parentCentre = Vector3(root.position.x - halfSide, root.position.y - halfSide, root.position.z - halfSide); //pos is centre +half side in pos; centre = pos -halfside

	//create 8 nodes
	for (int i = 0; i < 8; i++)
	{
		root.children.push_back(new Node());
		root.children[i]->sideLegnth = halfSide;
		root.children[i]->FindLocalPosition(i, parentCentre);

	}

	if (parentCentre.x == 0.0f || parentCentre.y == 0.0f || parentCentre.z == 0.0f)
	{
		parentCentre += Vector3(0.01f, 0.01f, 0.01f); //alter to avoid dividing by 0
	}


	//assign all particles to appropriate node
	for (int i = 0; i < root.particleCount; i++)
	{
		bool inside = true; // if particle is not inside extents of root then do not recursively split it.
		if (abs(root.particles[i]->position.x) > root.sideLegnth * 0.5f || abs(root.particles[i]->position.y) > root.sideLegnth * 0.5f || abs(root.particles[i]->position.z) > root.sideLegnth * 0.5f)
		{
			inside = false; //if not inside extents, set inside to false
		}
		//pos -centre point to find if coordinates are - or + directions from centre
		Vector3 dir = root.particles[i]->position - parentCentre;

		if (inside) //only place particle if inside extents(otherwise will not correctly place, and will inifinitely run until out of memory crash.
		{

			dir = Vector3((dir.x / abs(dir.x)), (dir.y / abs(dir.y)), (dir.z / abs(dir.z)));
			bool placed = false;
			int j = 0;
			while (!placed)
			{
				if (dir.equals(root.children[j]->localPosition))
				{
					root.children[j]->particles.push_back(root.particles[i]);
					root.children[j]->particleCount++;
					placed = true;
				}
				j++;
			}
		}



	}
}




void BHParallelTree::DoFoo()
{
	//int* f;

	//cudaMalloc(&f, 2 * sizeof(int));

	//printf("hello Dofoo");
	//AllPairs << <1, 2 >> > (n, particles*, timeStep);

	//cudaFree(f);

}

