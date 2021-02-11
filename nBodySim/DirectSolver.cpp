#include "DirectSolver.h"


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

void DirectSolver::SolveEuler(float dt, std::vector<Particle*>* particles, float timeStep)
{
	
	
	//update vel every time step
	//loop all particles
	for (int i = 0; i < particles->size(); i++)
	{
		//loop all particles 
		for (int j = 0; j < particles->size(); j++)
		{
			//if j and i are not same particle then calc j gravitational effect on i
			if (j != i)
			{
				//diff in positions
				Vector3 diff = particles->at(i)->position - particles->at(j)->position;
				float dist = diff.length(); //get distance

				float mult = (g * particles->at(j)->mass) / (dist * dist * dist); //multiplier  (g * mass )/ (distance ^3)

				Vector3 multDiff = Vector3(mult * diff.getX(), mult * diff.getY(), mult * diff.getZ()); //multiply  vector by multiplier to get force
				multDiff.scale(timeStep);
				particles->at(i)->velocity = particles->at(i)->velocity - multDiff; //new V = old v + acceleration due to gravity
				int f = 0;
				f = 1;
			}
		}
		particles->at(i)->Update();//update particles with new forces

	}

}

void DirectSolver::SolveRK4(float dt, std::vector<Particle*>* particles, float timeStep)
{
	



	//update vel every time step
	//loop all particles
	for (int i = 0; i < particles->size(); i++)
	{
		//loop all particles 
		for (int j = 0; j < particles->size(); j++)
		{
			//if j and i are not same particle then calc j gravitational effect on i
			if (j != i)
			{

				Vector3 diff = particles->at(i)->position - particles->at(j)->position;
				float dist = diff.length(); //get distance
				float mult = (g * particles->at(j)->mass) / (dist * dist * dist); //multiplier  (g * mass )/ (distance ^3)

				


				//k1 - euler acc
				Vector3 k1 = Vector3(mult * diff.getX(), mult * diff.getY(), mult * diff.getZ()); //multiply  vector by multiplier to get force

				//k2 - acc at 0.5 timesteps, based on k1
				k1.scale(0.5f); 							   
				Vector3 tempV = particles->at(i)->velocity + k1;
				tempV.scale(0.5f * timeStep);
				Vector3 tempP = particles->at(i)->position + tempV;
				Vector3 k2 = Vector3(particles->at(j)->position - tempP);
				k2.scale(mult);

				//k3 - acc at 0.5 timesteps, based on k2
				k2.scale(0.5f);
				tempV = particles->at(i)->velocity + k2;
				tempV.scale(0.5f * timeStep);
				tempP = particles->at(i)->position + tempV;
				Vector3 k3 = Vector3(particles->at(j)->position - tempP);
				k3.scale(mult);

				//k4 - location 1 timestep using k3 acc
				tempV = particles->at(i)->velocity + k3;
				tempV.scale(0.5f * timeStep);
				tempP = particles->at(i)->position + tempV;
				Vector3 k4 = Vector3(particles->at(j)->position - tempP);
				k4.scale(mult);

				k2.scale(2.0f);
				k3.scale(2.0f);
				Vector3 acc = k1 + k2;
				acc = acc + k3;
				acc = acc+ k4;
				acc.scale(1.0f / 6.0f);
				particles->at(i)->velocity += acc;

			}
			

		}
		particles->at(i)->Update();//update particles with new forces
	}

}
