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


#include <vector>
#include <sstream>
#include <string>

enum METHOD
{
	BH,
	EULER,
	RK4,
	VERLET
};

class Scene
{
	
private:
	
	ParticleManager* particleManager;
	Camera* camera;
	Input* input; 

	
	// For Window calculation.
	int width, height;
	float fov, ratio, nearPlane, farPlane;
	float time = 1.0f; // set so will be calculated at start
	
	float g = 6.67408e-11f; //grav constant
	float timeStep = 0.5f;

	//simulation paramaters, set to some default values in case read file fails
	int updates = 0;
	int newN = 2;
	int methodNext = 0;
	int method =0;
	int runFor = 100;
	int runForNext =100;
	float theta = 0.5f;
	float zoom;

	//for ui
	std::string methodText[5];
	std::string timeStepText;
	std::string thetaText;
	std::string runForText;
	std::string nText;


	

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

	void Restart();
	void ReadSetupFiles();
	void RenderString(float x, float y, std::string string);
	void InitCamera();
};

