#include "EulerSolver.h"


EulerSolver::EulerSolver(float grav)
{
	g = grav;
}

void EulerSolver::Solve(Particle* particles, float timeStep, int n)
{
	//update vel every time step
	//loop all particles

	for (int i = 0; i < n; i++)
	{
		//loop all particles 
		Vector3 acc = {};
		for (int j = 0; j < n; j++)
		{
			//if j and i are not same particle then calc j gravitational effect on i
			if (j != i)
			{
				acc = acc + CalculateAcceleration(particles[i].position, &particles[j]);
			}
		}

		//v(t+1) = v(t) +a(t) *dt
		acc.scale(timeStep);
		particles[i].velocity += acc;
		//x(t+1) = x(t) + v(t)*dt
		Vector3 vDt = particles[i].velocity;
		vDt.scale(timeStep);
		particles[i].nextPosition = particles[i].position + vDt;

	}
}
