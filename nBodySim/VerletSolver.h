#pragma once
#include "Solver.h"
class VerletSolver :
	public Solver
{
public:
	VerletSolver(float grav);
	void Solve(Particle* particles, float timeStep, int n);
};

