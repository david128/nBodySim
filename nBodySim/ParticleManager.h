#pragma once
#include "Particle.h"
#include <vector>
#include "Vector3.h"

class ParticleManager
{
public:
	ParticleManager(Vector3 extents);

	Particle* CreateRandomParticle();
	void AddParticle(Particle* part);
	void InitSystem(int hugeParticles, int largeParticles, int mediumParticles, int smallParticles );
	void InitSystem(int numParticles);

	Particle* GetParticle(int id);
	std::vector<Particle*>* GetParticles();

private:

	Vector3 systemExtents;

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

