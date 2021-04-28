#pragma once
#include "Particle.h"
#include <vector>
#include "Vector3.h"
#include "DirectSolver.h"
#include "BHTree.h"

#include "Direct.cuh"


#include "RK4Solver.h"
#include "EulerSolver.h"
#include "VerletSolver.h"
#include <string>


class ParticleManager
{
public:
	ParticleManager(Vector3 extents, float g, int numberOfParticles, int rf, std::string mN);
	~ParticleManager();

	Particle* CreateRandomParticle();
	void AddParticle(Particle* part);

	void InitSystem();
	void InitTestSystem();
	void InitDiskSystem(float minR, float maxR, float height);

	void InitMethod(int m);

	Particle* GetParticlesArray();
	



	void Update(float dt, float timeStep);
	float SumEnergy();
	void Reset();

	int n;
private:
	
	int sum = 0;
	int ran;
	int runFor;
	std::string methodName;

	float grav;
	Solver* solver;

	//particles
	Particle* particlesArray;
	
	float FindRandomSize(int min, int max);
	Vector3 FindRandomPos();
	Vector3 FindRandomVel(int maxSpeed);
	float FindVolume(float radius);

	std::vector<float> recordEnergy;
	std::vector<float> recordTime;

	void PrintResults(float timeStep);

};

