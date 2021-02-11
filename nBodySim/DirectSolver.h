#pragma once
#include "vector3.h"
#include <vector>
#include "Particle.h"

class DirectSolver
{

private:

	float time;
	float g;

public:

	DirectSolver(float gravConst);
	bool Update(float dt, float timeStep);
	void SolveEuler(float dt, std::vector<Particle*>* particles, float timeStep);
	void SolveRK4(float dt, std::vector<Particle*>* particles, float timeStep);

};

