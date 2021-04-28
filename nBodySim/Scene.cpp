#include "Scene.h"
#include <iostream>
#include <fstream>
#include <string>


Scene::Scene(Input *inp)
{
	input = inp;

	methodText[0] = "Euler";
	methodText[1] = "RK4";
	methodText[2] = "Verlet";
	methodText[3] = "Barnes_Hut";
	methodText[4] = "Euler GPU";

	ReadSetupFiles();
	
	//OpenGL settings
	glShadeModel(GL_SMOOTH);							// Enable Smooth Shading
	glClearColor(0.39f, 0.58f, 93.0f, 1.0f);			// Cornflour Blue Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glClearStencil(0);									// Clear stencil buffer
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations
	glEnable(GL_TEXTURE_2D);							//	enable textures
	glDepthFunc(GL_LEQUAL);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	//blend function settings



	particleManager = new ParticleManager(Vector3(10000.0f, 10000.0f, 10000.0f), g, newN, runFor, methodText[method]);

	//particleManager->InitSystem();
	particleManager->InitDiskSystem(500,500+ newN * 20,100);
	//particleManager->InitTestSystem();

	particleManager->InitMethod(method);
	InitCamera();
	
}

void Scene::InitCamera()
{
	//set camera 

	zoom = 2000 + newN * 40;
	
	camera = new Camera();
	camera->setXzAngle(90.0f);
	camera->setCameraLook(Vector3(0.0f, 0.0f, 0.0f));
	camera->setCameraPos(Vector3(0.0f, 0.0f, zoom));
	camera->setCameraUp(Vector3(0, 1, 0));
	camera->SetDistanceToLook(zoom);

}

void Scene::render(float dt)
{
	// Clear Color and Depth Buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	gluLookAt(camera->getCameraPos().getX(), camera->getCameraPos().getY(), camera->getCameraPos().getZ(), camera->getCameraLook().getX(), camera->getCameraLook().getY(), camera->getCameraLook().getZ(), camera->getCameraUp().getX(), camera->getCameraUp().getY(), camera->getCameraUp().getZ());

	nText = ("N :" + std::to_string(newN));
	timeStepText = ("Time Step:" + std::to_string(timeStep));
	runForText = ("Run For:" + std::to_string(runFor));
	thetaText = ("Theta :" + std::to_string(theta));
	
	float x = -1000 * zoom / 2000 /ratio;
	float y = 1000 * zoom / 2000 /ratio;
	

	RenderString(x, y,methodText[method]);
	RenderString(x, y* 0.9,nText);
	RenderString(x, y* 0.8, timeStepText);
	RenderString(x, y* 0.7, runForText);
	RenderString(x, y* 0.6, thetaText);

	
	


	Particle* particles = particleManager->GetParticlesArray();
	for (int i = 0; i < particleManager->n; i++)
	{
		particles[i].DrawParticle();
	}



	glutSwapBuffers();

}

void Scene::RenderString(float x, float y, std::string string)
{
	glRasterPos2f(x,y);
	for (int i = 0; i < string.size(); i++)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
	}


}

void Scene::handleInput(float dt)
{

	if (input->isKeyPressed('r') || input->isKeyPressed('R'))
	{
		Restart();
	}

	if (input->isKeyPressed('b') || input->isKeyPressed('B'))
	{
		if (newN >2)
		{
			newN--;
		}
	}
	if (input->isKeyPressed('n') || input->isKeyPressed('N'))
	{
		newN++;
	}

	if (input->isKeyPressed('m') || input->isKeyPressed('M'))
	{
		if (method == 4)
		{
			method = 0;
		}
		else
		{
			method++;
		}
	}

	if (input->isKeyPressed('l') || input->isKeyPressed('L'))
	{
		timeStep += 0.1;
	}
	if (input->isKeyPressed('k') || input->isKeyPressed('K'))
	{
		if (timeStep > 0)
		{
			timeStep -= 0.1;
		}
		
	}

	if (input->isKeyPressed('o') || input->isKeyPressed('O'))
	{
		if (theta >= 0.1)
		{
			theta -= 0.1;
		}

	}
	if (input->isKeyPressed('p') || input->isKeyPressed('P'))
	{
		if (theta < 1)
		{
			theta += 0.1;
		}

	}

	if (input->isKeyPressed('o') || input->isKeyPressed('O'))
	{
		if (theta >= 0.1)
		{
			theta -= 0.1;
		}

	}
	if (input->isKeyPressed('w') || input->isKeyPressed('W'))
	{
		runFor ++;
	}
	if (input->isKeyPressed('q') || input->isKeyPressed('Q"'))
	{
		if (runFor >1)
		{
			runFor--;
		}
		
	}

}

void Scene::update(float dt)
{

	//time = time + frame time
	time += dt;
	updates++;

	particleManager->Update(dt, timeStep);

	//update camera
	//camera->update();
	
}

void Scene::resize(int w, int h)
{
	width = w;
	height = h;
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0)
		h = 1;

	ratio = (float)w / (float)h;
	fov = 45.0f;
	nearPlane = 0.1f;
	farPlane = 10000000.0f;

	// Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);

	// Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(fov, ratio, nearPlane, farPlane);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);


}

void Scene::Restart()
{
	particleManager->~ParticleManager();
	particleManager = new ParticleManager(Vector3(10000.0f, 10000.0f, 10000.0f), g, newN, runFor, methodText[method]);
	particleManager->InitDiskSystem(500, 500 + newN * 10, 100);

	time = 0;
	updates = 0;

	particleManager->InitMethod(method);
	InitCamera();
}

void Scene::ReadSetupFiles()
{
	std::ifstream file("setup.txt");
	std::string temp, tempValue;

	std::string line;
	while (std::getline(file, line))
	{
		std::istringstream iss(line);
		if (!(iss >> temp >> tempValue)) { break; } // error
		if (temp[0] == 'M')//method
		{
			method = std::stoi(tempValue);
		}
		else if (temp[0] == 'N')//N
		{
			newN = std::stoi(tempValue);
		}
		else if (temp[1] == 'I')//Timestep
		{
			timeStep = std::stof(tempValue);
		}
		else if (temp[0] == 'R')
		{
			runFor = std::stoi(tempValue);
		}
		else if (temp[1] == 'H')//Theta
		{
			theta = std::stof(tempValue);
		}
	}
}

