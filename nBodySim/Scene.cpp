#include "Scene.h"
#include <iostream>
#include <fstream>
#include <string>
//temp




Scene::Scene(Input *inp)
{
	input = inp;

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



	particleManager = new ParticleManager(Vector3(10000.0f, 10000.0f, 10000.0f), g, 6);
	//particleManager->InitSystem();
	particleManager->InitDiskSystem(1500,4000,100);
	//particleManager->InitTestSystem();

	particleManager->InitMethod();


	InitCamera();
	
}

void Scene::InitCamera()
{
	//set camera 
	camera = new Camera();
	camera->setXzAngle(90.0f);
	camera->setCameraLook(Vector3(0.0f, 0.0f, 0.0f));
	camera->setCameraPos(Vector3(0.0f, 0.0f, 20000));
	camera->setCameraUp(Vector3(0, 1, 0));
	camera->SetDistanceToLook(20000.0f);

}

void Scene::render(float dt)
{
	// Clear Color and Depth Buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	gluLookAt(camera->getCameraPos().getX(), camera->getCameraPos().getY(), camera->getCameraPos().getZ(), camera->getCameraLook().getX(), camera->getCameraLook().getY(), camera->getCameraLook().getZ(), camera->getCameraUp().getX(), camera->getCameraUp().getY(), camera->getCameraUp().getZ());
	Particle* particles = particleManager->GetParticlesArray();
	for (int i = 0; i < particleManager->n; i++)
	{
		particles[i].DrawParticle();
	}


	glutSwapBuffers();

}

void Scene::handleInput(float dt)
{
	if (input->isKeyDown('d') || input->isKeyDown('D'))
	{
		camera->PanCamera(-0.2f *dt);
	}

	if (input->isKeyDown('a') || input->isKeyDown('A'))
	{
		camera->PanCamera(0.2f * dt);
	}


	if (input->isKeyDown('i')|| input->isKeyDown('I'))
	{
		camera->ZoomCamera(-100.0f * dt);
	}


	if (input->isKeyDown('o') || input->isKeyDown('O'))
	{
		camera->ZoomCamera(+100.0f * dt);
	}
}

void Scene::update(float dt)
{

	//time = time + frame time
	time += dt;
	updates++;

	particleManager->Update(dt, timeStep);
	if (updates == runFor)
	{
		std::ofstream outfile("output.txt");

		outfile << "ran in" << std::endl;

	}
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

	float ratio = (float)w / (float)h;
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

void Scene::ReadSetupFiles()
{
	std::ifstream file("setup.txt");
	std::string a,b,c,d;



	while (file >> a >> b >> c >> d);
	{
		method = 0;
	}
}
