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

void DirectSolver::Solve(float dt, std::vector<Particle*>* particles)
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
				particles->at(i)->velocity = particles->at(i)->velocity - multDiff; //new V = old v + acceleration due to gravity

			}
		}
		particles->at(i)->Update();//update particles with new forces

	}

}
