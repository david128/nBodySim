#include "ParticleManager.h"
#include <time.h> 

#include "cuda_runtime.h"
#include "cuda.h"


ParticleManager::ParticleManager(Vector3 extents, float g, int numberOfParticles)
{
	posSystemExtents = extents;
	negSystemExtents = extents;
	negSystemExtents.scale(-1.0f);
	direct = new DirectSolver(g);
	barnesHut = new BHTree(extents.x*10.0f, g, 0.5f);

	grav = g;
	n = numberOfParticles;

	int bytes = n * sizeof(Particle);

	cudaMallocManaged(&particlesArray, bytes);

	solver = new VerletSolver(g);
	solver = barnesHut;

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
	//particles.push_back(part);
}
	

void ParticleManager::InitSystem()
{
	
	srand(time(NULL));
	for (int i = 0; i < n; i++)
	{
		//create and store particle
		particlesArray[i] = *CreateRandomParticle();
	}

}	

void ParticleManager::InitTestSystem()
{
	Particle* p1 = new Particle(10, Vector3(0.0f, 0.0f, 0.0f), 60000000000000, Vector3(0.0f, 0.0f, 0.0f));
	Particle* p2 = new Particle(5, Vector3(0.0f, 35.0f, 0.0f), 50, Vector3(10.5f, 0.0f, 0.0f));
	particlesArray[0] = *p1;
	particlesArray[1] = *p2;


}

void ParticleManager::InitDiskSystem(float minR,float maxR, float height)
{
	Particle* newP; 
	float largeMass = 100000.0;
	float smallMass = 1;
	newP = new Particle(100, Vector3(0.0f, 0.0f, 0.0f),largeMass, Vector3(0.0f, 0.0f, 0.0f));
	particlesArray[0] = *newP;

	for (int i = 1; i < n; i++)
	{

		//theta = random?
		float theta = FindRandomSize(0, 360);
		//random radius
		float r = FindRandomSize(minR,maxR);
		float h = FindRandomSize(-height,height);
		float v = smallMass * sqrt((grav * largeMass) / r);

		newP = new Particle(50, Vector3(r*cosf(theta), r * sinf(theta),height), smallMass = 10, Vector3(v*sinf(theta), -v*cosf(theta), 0.0f));
		particlesArray[i] = *newP;

	}
}

Particle* ParticleManager::GetParticlesArray()
{
	return particlesArray;
}

void ParticleManager::Update(float dt, float timeStep)
{

	//if (direct->Update(dt, 0.5f))
	//{
		//direct->SolveEuler(dt, particlesArray, timeStep, n);
		//UpdateAllParticles(timeStep);
	///}

	//if (direct->Update(dt, 0.5f))
	//{
	//	direct->SolveVerlet(dt, particlesArray, timeStep,  n);
	//	UpdateAllParticles(timeStep);
	//}
	//	
	//if (barnesHut->Update(dt, timeStep))
	//{
	//	barnesHut->DeleteTree();
	//	barnesHut->ConstructTree(&particles);
	//	barnesHut->CalculateForces(0.5f, &particles,timeStep);
	//	UpdateAllParticles(timeStep);
	//}

	//if (direct->Update(dt, 0.5f))
	//{
	//	//printf("hello pM");
	//	parallelBarnesHut->DoFoo(n, particlesArray);
	//	UpdateAllParticles(timeStep);

	//}


	//if (direct->Update(dt, timeStep))
	//{
	//	direct->SolveRK4(dt, particlesArray, 0.1, n);
	//	UpdateAllParticles(timeStep);
	//}

	//if (solver->Update(dt, timeStep))
	{
		solver->Solve(particlesArray, 0.5, n);
		UpdateAllParticles(timeStep);
	}

}

void ParticleManager::UpdateAllParticles(float timeStep)
{
	for (int i = 0; i < n; i++)
	{
		particlesArray[i].Update(timeStep);
		
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

