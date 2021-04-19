#include "RK4Solver.h"

RK4Solver::RK4Solver(float grav)
{
	g= grav;
}

void RK4Solver::Solve(Particle* particles, float timeStep, int n)
{
	std::vector<Vector3> kx1, kx2, kx3, kx4, kv1, kv2, kv3, kv4;

	kx1.resize(n);
	kx2.resize(n);
	kx3.resize(n);
	kx4.resize(n);

	kv1.resize(n);
	kv2.resize(n);
	kv3.resize(n);
	kv4.resize(n);


	//k1
	for (int i = 0; i < n; i++)
	{
		//loop all particles 
		for (int j = 0; j < n; j++)
		{
			//if j and i are not same particle then calc j gravitational effect on i
			if (j != i)
			{
				//kx1 = h *v(n)
				kx1[i] = particles[i].velocity;
				kx1[i].scale(timeStep);

				//kv1 = h*(x(n) - x[j](n)) * (g * m[j] / |x(n) - x[j](n)|^3)
				kv1[i] = CalculateAcceleration(particles[i].position, &particles[j]);
				kv1[i].scale(timeStep);
			}

		}
	}

	//k2
	for (int i = 0; i < n; i++)
	{
		//loop all particles 
		for (int j = 0; j < n; j++)
		{
			//if j and i are not same particle then calc j gravitational effect on i
			if (j != i)
			{
				//kx2 = h*(v(n) +kv1/2)	
				kx2[i] = kv1[i];
				kx2[i].scale(0.5f);
				kx2[i] = particles[i].velocity + kx2[i];
				kx2[i].scale(timeStep);

				//kv2 = h * ((x(n) + kx1/2) - x[j](n)) * (g * m[j] / |(x(n) + kx1/2) - x[j](n) | ^ 3)
				kv2[i] = kx1[i];
				kv2[i].scale(0.5f);
				kv2[i] = particles[i].position + kv2[i];
				kv2[i] = CalculateAcceleration(kv2[i], &particles[j]);
				kv2[i].scale(timeStep);
			}

		}
	}

	//k3
	for (int i = 0; i < n; i++)
	{
		//loop all particles 
		for (int j = 0; j < n; j++)
		{
			//if j and i are not same particle then calc j gravitational effect on i
			if (j != i)
			{
				//kx3 = h*(v(n) +kv2/2)	
				kx3[i] = kv2[i];
				kx3[i].scale(0.5f);
				kx3[i] = particles[i].velocity + kx3[i];
				kx3[i].scale(timeStep);

				//kv3 = h * ((x(n) + kx2/2) - x[j](n)) * (g * m[j] / |(x(n) + kx2/2) - x[j](n) | ^ 3)
				kv3[i] = kx2[i];
				kv3[i].scale(0.5f);
				kv3[i] = particles[i].position + kv3[i];
				kv3[i] = CalculateAcceleration(kv3[i], &particles[j]);
				kv3[i].scale(timeStep);
			}

		}
	}


	//k4
	for (int i = 0; i < n; i++)
	{
		//loop all particles 
		for (int j = 0; j < n; j++)
		{
			//if j and i are not same particle then calc j gravitational effect on i
			if (j != i)
			{
				//kx4 = h*(v(n) +kv3)	
				kx4[i] = particles[i].velocity + kx3[i];
				kx4[i].scale(timeStep);

				//kv2 = h * ((x(n) + kx3) - x[j](n)) * (g * m[j] / |(x(n) + kx3) - x[j](n) | ^ 3)
				kv4[i] = particles[i].position + kx3[i];
				kv4[i] = CalculateAcceleration(kv4[i], &particles[j]);
				kv4[i].scale(timeStep);
			}

		}
	}

	//x and v
	for (int i = 0; i < n; i++)
	{

		kx1[i].scale(1.0f / 6.0f);
		kx2[i].scale(1.0f / 3.0f);
		kx3[i].scale(1.0f / 3.0f);
		kx4[i].scale(1.0f / 6.0f);

		kv1[i].scale(1.0f / 6.0f);
		kv2[i].scale(1.0f / 3.0f);
		kv3[i].scale(1.0f / 3.0f);
		kv4[i].scale(1.0f / 6.0f);

		particles[i].nextPosition = particles[i].position + kx1[i] + kx2[i] + kx3[i] + kx4[i];
		particles[i].velocity = particles[i].velocity + kv1[i] + kv2[i] + kv3[i] + kv4[i];

	}
}
