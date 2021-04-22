#pragma once
#include "vector3.h"
#include <vector>
#include "Particle.h"

class DirectSolver
{

private:

	float time;
	float g;
	Vector3 CalculateAcceleration(Vector3 posI, Vector3 posJ, float mass);

	
public:

	DirectSolver(float gravConst);
	bool Update(float dt, float timeStep);
	void SolveEuler(float dt, Particle* particles, float timeStep, int n);
	void SolveRK4(float dt, Particle* particles, float timeStep, int n);
	void SolveVerlet(float dt, Particle* particles, float timeStep, int n);
	
};

