#include "VerletSolver.h"

VerletSolver::VerletSolver(float grav)
{
	g = grav;
}

void VerletSolver::Solve(Particle* particles, float timeStep, int n)
{
	for (int i = 0; i < n; i++)
	{
		//new pos = pos(t) + (currentV * h ) + (0.5 * current a * h^2)
		Vector3 currentVdt = particles[i].velocity;
		currentVdt.scale(timeStep);

		Vector3 halfADt2 = particles[i].acceleration;
		halfADt2.scale(0.5f * timeStep * timeStep);

		particles[i].nextPosition = particles[i].position + currentVdt + halfADt2;
		particles[i].position = particles[i].position + currentVdt + halfADt2;
	}

	for (int i = 0; i < n; i++)
	{
		Vector3 halfAccDt = particles[i].acceleration;
		halfAccDt.scale(0.5f * timeStep);
		particles[i].velocity += halfAccDt;
	}

	for (int i = 0; i < n; i++)
	{
		Vector3 acc = {};
		//loop all particles 
		for (int j = 0; j < n; j++)
		{
			//if j and i are not same particle then calc j gravitational effect on i
			if (j != i)
			{
				//sum accelerations
				acc = acc + CalculateAcceleration(particles[i].position, particles[j].position, particles[j].mass);
			}

		}

		particles[i].acceleration = acc;
	}


	for (int i = 0; i < n; i++)
	{
		Vector3 halfAccDt = particles[i].acceleration;
		halfAccDt.scale(0.5f * timeStep);
		particles[i].velocity += halfAccDt;
	}
}
