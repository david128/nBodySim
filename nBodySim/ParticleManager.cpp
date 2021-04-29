#include "ParticleManager.h"
#include <time.h> 
#include "cuda_runtime.h"
#include "cuda.h"
#include <chrono>
#include <iostream>
#include <fstream>


ParticleManager::ParticleManager(Vector3 extents, float g, int numberOfParticles, int rf, std::string mN)
{
	grav = g;
	n = numberOfParticles;
	ran = 0;
	runFor = rf;
	methodName = mN;

	int bytes = n * sizeof(Particle);
	cudaMallocManaged(&particlesArray, bytes);
	
}

ParticleManager::~ParticleManager()
{
	recordEnergy.clear();
	recordTime.clear();
	delete solver;
	solver = nullptr;
	//free from memory
	cudaFree(particlesArray);
}

Particle* ParticleManager::CreateRandomParticle()
{
	
	//create new particle
	Particle* p = new Particle();
	p->size = FindRandomSize(50, 100); //set size between 50 and 300
	p->mass = FindVolume(p->size); // set mass to volume
	p->position = FindRandomPos();
	p->velocity = FindRandomVel(50);
	
	return p;
}

//places particles in random places
void ParticleManager::InitSystem()
{
	srand(time(NULL));
	for (int i = 0; i < n; i++)
	{
		//create and store particle
		particlesArray[i] = *CreateRandomParticle();
	}

}	

//places particles in random disk system
void ParticleManager::InitDiskSystem(float minR,float maxR, float height)
{
	Particle* newP; 
	float largeMass = 1000000000000000000.0; //large mass
	float smallMass = 1; //mass of smaller 
	newP = new Particle(100, Vector3(0.0f, 0.0f, 0.0f),largeMass, Vector3(0.0f, 0.0f, 0.0f));
	particlesArray[0] = *newP; //large mass particles orbit round

	for (int i = 1; i < n; i++)
	{

		//theta = random float
		float theta = FindRandomFloat(0.0f,360.0f);
		//random radius and h
		float h = FindRandomFloat(-height, height);
		float r = FindRandomFloat(minR,maxR);
		float v =  sqrt((grav * largeMass) / r);

		newP = new Particle(50, Vector3(r*cosf(theta), r * sinf(theta),height), smallMass = 10, Vector3(v*sinf(theta), -v*cosf(theta), 0));
		particlesArray[i] = *newP;

	}
}

void ParticleManager::InitMethod(int m, float th)
{
	//passed an int and chooses how to solve
	if (m ==1)
	{
		solver = new RK4Solver(grav);
	}
	else if (m== 2)
	{
		solver = new VerletSolver(grav);
	}
	else if (m ==3)
	{
		theta = th;
		solver = new BHTree(grav, theta, particlesArray, n);
	}
	else if (m == 4)
	{
		solver = new DirectGPU(n);
	}
	else
	{
		solver = new EulerSolver(grav);
	}

}

Particle* ParticleManager::GetParticlesArray()
{
	return particlesArray;
}

void ParticleManager::Update(float dt, float timeStep)
{
	//sum energy to check accuracy
	if (ran == sum * 10)
	{
		recordEnergy.push_back(SumEnergy());
		sum++;
	}

	//record time
	auto start = std::chrono::high_resolution_clock::now();
	//solve
	solver->Solve(particlesArray, timeStep, n);
	auto stop = std::chrono::high_resolution_clock::now();

	//find time
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout<< "Run "<< ran << " ran in "<<  duration.count() << " microseconds."<<  std::endl;


	ran++;
	recordTime.push_back(duration.count());
	if (ran == runFor) 
	{
		PrintResults(timeStep);
	}
	
}

float ParticleManager::SumEnergy()
{
	//sums PE + KE to find total energy of system which should be constant
	float energy = 0;
	float velocity;
	float dist;
	for (int i = 0; i < n; i++)
	{
		//KE
		velocity = particlesArray[i].velocity.length();
		energy += 0.5 * particlesArray[i].mass * velocity * velocity;
		for (int j = 0; j < n; j++)
		{
			if (i!=j)
			{
				//PE
				dist = Vector3(particlesArray[j].position - particlesArray[i].position).length();
				energy += grav * particlesArray[i].mass * particlesArray[j].mass / dist;
			}
		}
	}
	return energy;
}
	




float ParticleManager::FindRandomSize(int min, int max)
{
	//srand(time(NULL));
	return float(rand() % max + min);  //rand size between 50 and 300
}

float ParticleManager::FindRandomFloat(float min, float max) {
	return min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
}

Vector3 ParticleManager::FindRandomPos()
{

	///return pos in extents of system
	Vector3 pos = Vector3(float(rand() % (int)(1000 * 2.0f) + (int)-1000),
		float(rand() % (int)(1000 * 2.0f) + (int)-1000),
		float(rand() % (int)(1000 * 2.0f) + (int)-1000));

	return pos;
}

Vector3 ParticleManager::FindRandomVel(int maxSpeed)
{
	//find random direction
	float x = -1.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));
	float y = -1.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));
	float z = -1.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));

	Vector3 direction = { x,y,z };
	direction.normalise(); //normalise to create direction vector
	float speed = float(rand() % maxSpeed * 2 - maxSpeed); //find random speed

	direction = { direction.x * speed, direction.y * speed, direction.z * speed }; 
	return direction; //return direction * speed
}

float ParticleManager::FindVolume(float radius)
{
	return 4.0f/3.0f * 3.14159 * radius * radius* radius ; //volume of sphere
}

void ParticleManager::PrintResults(float timeStep)
{
	
	std::ofstream outFile;
	std::string fileName;
	std::string ts;
	std::string th = "";

	//add th if method is bh
	if (methodName == "Barnes_Hut")
	{
		th = "_theta_" + std::to_string(theta)  + "_";
		//convert . in ts to _ for file name
		for (int i = 0; i < th.size(); i++)
		{
			if (th[i] == '.')
			{
				th[i] = '_';
			}
		}
	}



	ts = std::to_string(timeStep);
	//convert . in ts to _ for file name
	for (int i = 0; i < ts.size(); i++)
	{
		if (ts[i] == '.')
		{
			ts[i] = '_';
		}
	}


	fileName = "./output/"+ std::to_string(n) +  "bodies_" + methodName + th+ "_TS_" + ts + "_RF_" + std::to_string(runFor) + "_times.csv";

	outFile.open(fileName);//open this file

	for (int i = 0; i < recordTime.size(); i++) //loop through each item in array and write to file
	{
		outFile << recordTime[i]<< "\n";
	}

	outFile.close();//close file


	fileName = "./output/" + std::to_string(n) + "bodies_" + methodName + "_TS_" + ts + "_RF_" + std::to_string(runFor) + "_energy.csv";
	outFile.open(fileName);//open this file


	for (int i = 0; i < recordEnergy.size(); i++) //loop through each item in array and write to file
	{
		outFile << recordEnergy[i]<< "\n";
	}

	outFile.close();//close file
	

}

void ParticleManager::Reset()
{
	recordEnergy.clear();
	recordTime.clear();

	//free from memory
	cudaFree(particlesArray);
	sum = 0;
	ran = 0;

}

