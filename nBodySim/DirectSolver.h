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
	void Solve(float dt, std::vector<Particle*>* particles);

};

