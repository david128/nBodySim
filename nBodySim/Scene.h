#pragma once

// Include
#include <GL/freeglut.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <stdio.h>
#include "Camera.h"
#include"Shape.h"
#include "Particle.h"

class Scene
{
private:
	Shape* shape;
	Particle* particle;
	Camera* camera;
	// For Window and frustum calculation.
	int width, height;
	float fov, nearPlane, farPlane;

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

