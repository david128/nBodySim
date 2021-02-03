#pragma once

// Include
#include <GL/freeglut.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <stdio.h>
#include "Camera.h"
#include "Input.h"
#include "Particle.h"
#include "ParticleManager.h"
#include "DirectSolver.h"

#include <vector>

class Scene
{
	
private:
	/*Particle* particle;
	Particle* particle2;
	Particle* particle3;
	Particle* particle4;
	std::vector<Particle*> particles;*/
	
	ParticleManager* particleManager;
	
	Camera* camera;
	Input* input; 

	DirectSolver* direct;

	// For Window and frustum calculation.
	int width, height;
	float fov, nearPlane, farPlane;

	float time = 1.0f; // set so will be calculated at start
	
	float g = 6.67408e-11f; //grav constant

	float timeStep = 0.5f;

public:
	Scene(Input* inp);
	// render function
	void render(float dt);
	// Handle input function
	void handleInput(float dt);
	// Update function 
	void update(float dt);
	//resize
	void resize(int w, int h);

	void InitCamera();
	void MoveCamera();
};

