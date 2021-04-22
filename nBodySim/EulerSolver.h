#pragma once
#include "Solver.h"
class EulerSolver :
	public Solver
{

public:
	EulerSolver(float grav);
	void Solve(Particle* particles, float timeStep, int n);
};

