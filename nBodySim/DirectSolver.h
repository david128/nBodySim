#pragma once
#include "ParticleManager.h"
class DirectSolver
{

private:

	float time;
	float g;
	ParticleManager* particleManager;
public:

	DirectSolver(ParticleManager* pm, float gravConst);
	void Update(float dt, float timeStep);
	void Solve(float dt);

};

