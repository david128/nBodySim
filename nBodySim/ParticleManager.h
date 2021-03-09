#pragma once
#include "Particle.h"
#include <vector>
#include "Vector3.h"
#include "DirectSolver.h"
#include "BHTree.h"
#include "BH.cuh"

class ParticleManager
{
public:
	ParticleManager(Vector3 extents, float g);

	Particle* CreateRandomParticle();
	void AddParticle(Particle* part);
	void InitSystem(int hugeParticles, int largeParticles, int mediumParticles, int smallParticles );
	void InitSystem(int numParticles);

	Particle* GetParticle(int id);
	std::vector<Particle*>* GetParticles();
	

	DirectSolver* direct;
	BHTree* barnesHut;
	BHParallelTree* parallelBarnesHut;

	void Update(float dt, float timeStep);
	void UpdateAllParticles(float timeStep);

private:

	Vector3 posSystemExtents;
	Vector3 negSystemExtents;

	//particles
	std::vector<Particle*> particles;
	Particle hugeParticle;
	Particle largeParticle;
	Particle mediumParticle;
	Particle smallParticle;
	
	float FindRandomSize(int min, int max);
	Vector3 FindRandomPos();
	Vector3 FindRandomVel(int maxSpeed);
	float FindVolume(float radius);
};

