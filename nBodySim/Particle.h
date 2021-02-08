#pragma once
#include "Vector3.h"
#include <GL\freeglut.h>
class Particle
{
public:
	Particle(float s, Vector3 p,float m); 
	Particle();
	~Particle();

	Vector3 position;
	float size;
	float mass;
	//Vector3 acceleration;
	Vector3 velocity;

	
	void DrawParticle();
	void Update();
};

