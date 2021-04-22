#include "DirectSolver.h"


Vector3 DirectSolver::CalculateAcceleration(Vector3 posI, Vector3 posJ, float mass)
{
	Vector3 diff = posI - posJ;
	float dist = diff.length(); //get distance
	float multiplier = (g * mass) / (dist * dist * dist); //multiplier  (g * mass )/ (distance ^3)
	Vector3 acc = diff;
	acc.scale(-multiplier); //return as negative as gravitational effect in negative
	return acc;
}



DirectSolver::DirectSolver( float gravConst)
{

	time = 0.0f;
	g = gravConst;
}

bool DirectSolver::Update(float dt, float timeStep)
{
	time += dt;

	if (time >= timeStep)
	{
		time = 0.0f;//reset time
		return true; //return true so we now solve
		
	}
	return false;
}

void DirectSolver::SolveEuler(float dt, Particle* particles, float timeStep, int n)
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
				acc = acc + CalculateAcceleration(particles[i].position, particles[j].position, particles[j].mass);
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

void DirectSolver::SolveRK4(float dt, Particle* particles, float timeStep, int n)
{
	std::vector<Vector3> kx1,kx2,kx3,kx4,kv1,kv2,kv3,kv4;

	kx1.resize(n);
	kx2.resize(n);
	kx3.resize(n);
	kx4.resize(n);
	
	kv1.resize(n);
	kv2.resize(n);
	kv3.resize(n);
	kv4.resize(n);


	
	//update vel every time step
	//loop all particles
	//for (int i = 0; i < n; i++)
	//{
	//	//loop all particles 
	//	for (int j = 0; j < n; j++)
	//	{
	//		//if j and i are not same particle then calc j gravitational effect on i
	//		if (j != i)
	//		{
	//			//kx1 = h *v(n)
	//			Vector3 kx1 = particles[i].velocity;
	//			kx1.scale(timeStep);

	//			//kv1 = h*(x(n) - x[j](n)) * (g * m[j] / |x(n) - x[j](n)|^3)
	//			Vector3 kv1 = CalculateAcceleration(particles[i].position, &particles[j]);
	//			kv1.scale(timeStep);

	//			//kx2 = h*(v(n) +kv1/2)	
	//			Vector3 kx2 = kv1;
	//			kx2.scale(0.5f);
	//			kx2 = particles[i].velocity + kx2;
	//			kx2.scale(timeStep);

	//			//kv2 = h * ((x(n) + kx1/2) - x[j](n)) * (g * m[j] / |(x(n) + kx1/2) - x[j](n) | ^ 3)
	//			Vector3 kv2 = kx1;
	//			kv2.scale(0.5f);
	//			kv2 = particles[i].position + kv2;
	//			kv2 = CalculateAcceleration(kv2, &particles[j]);
	//			kv2.scale(timeStep);

	//			//kx3 = h*(v(n) +kv2/2)	
	//			Vector3 kx3 = kv2;
	//			kx3.scale(0.5f);
	//			kx3 = particles[i].velocity + kx3;
	//			kx3.scale(timeStep);

	//			//kv3 = h * ((x(n) + kx2/2) - x[j](n)) * (g * m[j] / |(x(n) + kx2/2) - x[j](n) | ^ 3)
	//			Vector3 kv3 = kx2;
	//			kv3.scale(0.5f);
	//			kv3 = particles[i].position + kv3;
	//			kv3 = CalculateAcceleration(kv3, &particles[j]);
	//			kv3.scale(timeStep);

	//			//kx4 = h*(v(n) +kv3)	
	//			Vector3 kx4 = particles[i].velocity + kx3;
	//			kx4.scale(timeStep);

	//			//kv2 = h * ((x(n) + kx3) - x[j](n)) * (g * m[j] / |(x(n) + kx3) - x[j](n) | ^ 3)
	//			Vector3 kv4 = particles[i].position + kx3;
	//			kv4 = CalculateAcceleration(kv4, &particles[j]);
	//			kv4.scale(timeStep);

	//			kx1.scale(1.0f / 6.0f);
	//			kx2.scale(1.0f / 3.0f);
	//			kx3.scale(1.0f / 3.0f);
	//			kx4.scale(1.0f / 6.0f);

	//			kv1.scale(1.0f / 6.0f);
	//			kv2.scale(1.0f / 3.0f);
	//			kv3.scale(1.0f / 3.0f);
	//			kv4.scale(1.0f / 6.0f);

	//			particles[i].nextPosition = particles[i].position + kx1 + kx2 + kx3 + kx4;
	//			particles[i].velocity = particles[i].velocity + kv1 + kv2 + kv3 + kv4;
	//			
	//		}
	//		

	//	}

	//}

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
				kv1[i] = CalculateAcceleration(particles[i].position, particles[j].position, particles[j].mass);
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
				kv2[i] = CalculateAcceleration(kv2[i], particles[j].position, particles[j].mass);
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
				kv3[i] = CalculateAcceleration(kv3[i], particles[j].position, particles[j].mass);
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
				kv4[i] = CalculateAcceleration(kv4[i], particles[j].position, particles[j].mass);
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

void DirectSolver::SolveVerlet(float dt, Particle* particles, float timeStep, int n)
{

	for (int i = 0; i < n; i ++ )
	{
		//new pos = pos(t) + (currentV * h ) + (0.5 * current a * h^2)
		Vector3 currentVdt = particles[i].velocity;
		currentVdt.scale(timeStep);

		Vector3 halfADt2 = particles[i].acceleration;
		halfADt2.scale(0.5f * timeStep * timeStep);

		particles[i].nextPosition = particles[i].position + currentVdt + halfADt2;
		particles[i].position = particles[i].position + currentVdt + halfADt2;
	}

	for (int i = 0; i < n; i ++ )
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

	//for (int i = 0; i < n; i++)
	//{
	//	Vector3 acc = {};
	//	//loop all particles 
	//	for (int j = 0; j < n; j++)
	//	{
	//		//if j and i are not same particle then calc j gravitational effect on i
	//		if (j != i)
	//		{
	//			//sum accelerations
	//			acc = acc + CalculateAcceleration(particles[i].position, &particles[j]);
	//		}

	//	}

	//	//if (first[i] == true)
	//	//{
	//	//	particles[i].acceleration = acc;
	//	//	first[i] = false;
	//	//}

	//	////new pos = pos(t) + (currentV * h ) + (0.5 * current a * h^2)
	//	//Vector3 currentVdt = particles[i].velocity;
	//	//currentVdt.scale(timeStep);

	//	//Vector3 halfADt2 = particles[i].acceleration;
	//	//halfADt2.scale(0.5f * timeStep * timeStep);

	//	//particles[i].nextPosition = particles[i].position + currentVdt + halfADt2;

	//	////new v = v(t) + (old a + new a) * 0.5 * h
	//	//Vector3 oldPlusNewA = particles[i].acceleration + acc;
	//	//oldPlusNewA.scale(0.5f * timeStep);

	//	//particles[i].velocity = particles[i].velocity + oldPlusNewA;

	//	////store acc for next calc
	//	//particles[i].acceleration = acc;


	//	Vector3 velHalfStep = particles[i].acceleration;
	//	velHalfStep.scale(0.5f * timeStep);
	//	velHalfStep  = particles[i].velocity + velHalfStep;

	//	Vector3 newVHS = velHalfStep;
	//	newVHS.scale(timeStep);

	//	particles[i].nextPosition = particles[i].position + newVHS;

	//	particles[i].acceleration = acc;

	//	acc.scale(0.5f * timeStep);

	//	particles[i].velocity = velHalfStep + acc;

	//	


	//}
}
