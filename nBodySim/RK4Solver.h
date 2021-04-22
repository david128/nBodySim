#pragma once
#include "Solver.h"
class RK4Solver :
	public Solver
{
public:
	RK4Solver(float grav);

	void Solve(Particle* particles, float timeStep, int n);
};




	