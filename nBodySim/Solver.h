#pragma once

#include "vector3.h"
#include <vector>
#include "Particle.h"

class Solver
{
protected:
	float time;
	float g;
	Vector3 CalculateAcceleration(Vector3 posI, Vector3 posJ, float mass);
public:

	Solver();
	bool Update(float dt, float timeStep);
	virtual void Solve(Particle* particles, float timeStep, int n) = 0;
};

