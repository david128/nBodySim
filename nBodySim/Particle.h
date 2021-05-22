#pragma once
#include "Vector3.h"
#include <GL\freeglut.h>
class Particle
{
public:
	Particle(float s, Vector3 p,float m); 
	Particle(float s, Vector3 p,float m, Vector3 v); 
	Particle();
	

	Vector3 position;
	float size;
	float mass;
	Vector3 velocity;
	Vector3 acceleration;
	Vector3 colour = { 1.0f,1.0f,1.0f };
	
	void DrawParticle();
	void SetColour(Vector3 col);
private:
	
};

