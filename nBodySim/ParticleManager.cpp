#include "ParticleManager.h"
#include <time.h> 


ParticleManager::ParticleManager(Vector3 extents)
{
	systemExtents = extents;
}

Particle* ParticleManager::CreateRandomParticle()
{
	
	//create new particle
	Particle* p = new Particle();

	p->size = FindRandomSize(50, 100); //set size between 50 and 300
	p->mass = FindVolume(p->size); // set mass to volume
	p->position = FindRandomPos();
	p->velocity = FindRandomVel(300);
	
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



float ParticleManager::FindRandomSize(int min, int max)
{
	//srand(time(NULL));
	return float(rand() % max + min);  //rand size between 50 and 300
}



Vector3 ParticleManager::FindRandomPos()
{

	///return pos in extents of system
	Vector3 pos = Vector3(float(rand() % (int)(systemExtents.x * 2.0f) + (int)-systemExtents.x),
		float(rand() % (int)(systemExtents.y * 2.0f) + (int)-systemExtents.y),
		float(rand() % (int)(systemExtents.z * 2.0f) + (int)-systemExtents.z));

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

