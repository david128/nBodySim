#pragma once

// Include
#include <GL/freeglut.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <stdio.h>
#include "Camera.h"
#include"Shape.h"
#include "Particle.h"
#include <vector>

class Scene
{
private:
	Shape* shape;
	Particle* particle;
	Particle* particle2;
	Particle* particle3;
	std::vector<Particle*> particles;
	
	
	Camera* camera;
	// For Window and frustum calculation.
	int width, height;
	float fov, nearPlane, farPlane;

	float time = 11.0f;
	

public:
	Scene();
	// render function
	void render(float dt);
	// Handle input function
	void handleInput(float dt);
	// Update function 
	void update(float dt);
	//resize
	void resize(int w, int h);
};

