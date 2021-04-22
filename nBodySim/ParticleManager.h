#pragma once
#include "Particle.h"
#include <vector>
#include "Vector3.h"
#include "DirectSolver.h"
#include "BHTree.h"
#include "Direct.cuh"
#include "BarnesHut.cuh"

class ParticleManager
{
public:
	ParticleManager(Vector3 extents, float g, int numberOfParticles);

	Particle* CreateRandomParticle();
	void AddParticle(Particle* part);
	void InitSystem(int hugeParticles, int largeParticles, int mediumParticles, int smallParticles );
	void InitSystem();

	Particle* GetParticle(int id);
	Particle* GetParticlesArray();
	

	DirectSolver* direct;
	BHTree* barnesHut;
	BarnesHutGPU* parallelBarnesHut;
	DirectGPU* directGPU;

	void Update(float dt, float timeStep);
	void UpdateAllParticles(float timeStep);


	int n;
private:

	Vector3 posSystemExtents;
	Vector3 negSystemExtents;

	//particles

	Particle* particlesArray;

	Particle hugeParticle;
	Particle largeParticle;
	Particle mediumParticle;
	Particle smallParticle;
	
	float FindRandomSize(int min, int max);
	Vector3 FindRandomPos();
	Vector3 FindRandomVel(int maxSpeed);
	float FindVolume(float radius);

	bool first = true;

};

