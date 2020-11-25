#pragma once
#include "Vector3.h"
#include <GL\freeglut.h>
class Particle
{
public:
	Particle(float s, Vector3 p); 
	Particle();
	~Particle();

	float size;
	float mass;
	Vector3 acceleration;
	Vector3 velocity;
	Vector3 position;
	
	void DrawParticle();
	void Update(float dt);
};

