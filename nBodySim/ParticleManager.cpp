#include "ParticleManager.h"
#include <time.h> 


ParticleManager::ParticleManager(Vector3 extents, float g)
{
	posSystemExtents = extents;
	negSystemExtents = extents;
	negSystemExtents.scale(-1.0f);
	direct = new DirectSolver(g);
	barnesHut = new BHTree(extents.x*10.0f, g);

}

Particle* ParticleManager::CreateRandomParticle()
{
	
	//create new particle
	Particle* p = new Particle();

	p->size = FindRandomSize(50, 100); //set size between 50 and 300
	p->mass = FindVolume(p->size); // set mass to volume
	p->position = FindRandomPos();
	p->velocity = FindRandomVel(50);
	
	return p;
}

void ParticleManager::AddParticle(Particle* part)
{
	particles.push_back(part);
}
	

void ParticleManager::InitSystem(int numParticles)
{
	srand(time(NULL));
	for (int i = 0; i < numParticles; i++)
	{
		//create and store particle
		particles.push_back(CreateRandomParticle());
	}

}	

std::vector<Particle*>* ParticleManager::GetParticles()
{
	return &particles;
}

void ParticleManager::Update(float dt, float timeStep)
{

	//if (direct->Update(dt, 0.5f))
	//{
	//	direct->SolveEuler(dt, &particles, timeStep);
	//	UpdateAllParticles(timeStep);
	//}

	if (direct->Update(dt, 0.5f))
	{
		direct->SolveVerlet(dt, &particles, timeStep);
		UpdateAllParticles(timeStep);
	}
		
	//if (barnesHut->Update(dt, timeStep))
	//{
	//	barnesHut->DeleteTree();
	//	barnesHut->ConstructTree(&particles);
	//	barnesHut->CalculateForces(0.5f, &particles,timeStep);

	//}

	//
	//if (direct->Update(dt, timeStep))
	//{
	//	direct->SolveRK4(dt, &particles, timeStep);
	//	UpdateAllParticles(timeStep);
	//}

}

void ParticleManager::UpdateAllParticles(float timeStep)
{
	for (int i = 0; i < particles.size(); i++)
	{
		particles[i]->Update(timeStep);
		
	}
}



float ParticleManager::FindRandomSize(int min, int max)
{
	//srand(time(NULL));
	return float(rand() % max + min);  //rand size between 50 and 300
}



Vector3 ParticleManager::FindRandomPos()
{

	///return pos in extents of system
	Vector3 pos = Vector3(float(rand() % (int)(posSystemExtents.x * 2.0f) + (int)negSystemExtents.x),
		float(rand() % (int)(posSystemExtents.y * 2.0f) + (int)negSystemExtents.y),
		float(rand() % (int)(posSystemExtents.z * 2.0f) + (int)negSystemExtents.z));

	return pos;
}

Vector3 ParticleManager::FindRandomVel(int maxSpeed)
{
	//find random direction
	float x = -1.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));
	float y = -1.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));
	float z = -1.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));

	Vector3 direction = { x,y,z };
	direction.normalise(); //normalise to create direction vector
	float speed = float(rand() % maxSpeed * 2 - maxSpeed); //find random speed

	direction = { direction.x * speed, direction.y * speed, direction.z * speed }; 
	return direction; //return direction * speed
}

float ParticleManager::FindVolume(float radius)
{
	return 4.0f/3.0f * 3.14159 * radius * radius* radius ; //volume of sphere
}

