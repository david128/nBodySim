#include "Solver.h"

Vector3 Solver::CalculateAcceleration(Vector3 posI, Particle* pJ)
{
	Vector3 diff = posI - pJ->position;
	float dist = diff.length(); //get distance
	float multiplier = (g * pJ->mass) / (dist * dist * dist); //multiplier  (g * mass )/ (distance ^3)
	Vector3 acc = diff;
	acc.scale(-multiplier); //return as negative as gravitational effect in negative
	return acc;
}

Solver::Solver()
{
	time = 0.0f;

}

bool Solver::Update(float dt, float timeStep)
{
	time += dt;

	if (time >= timeStep)
	{
		time = 0.0f;//reset time
		return true; //return true so we now solve

	}
	return false;
}

