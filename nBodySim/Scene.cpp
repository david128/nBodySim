#include "Scene.h"


//temp
float angle = 0.0f;
float x;
float y;
float g = 10.0f;

Scene::Scene()
{
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
	// Other OpenGL / render setting should be applied here.
	
	shape = new Shape();
	particle = new Particle(10.0f, Vector3(-200.0f,50.0f,0.0f));

	particle2 = new Particle(20.0f, Vector3(0.0f, 0.0f, 0.0f));
	particle3 = new Particle(10.0f, Vector3(200.0f, -70.0f, 0.0f));
	


	particles.push_back(particle);
	particles.push_back(particle2);
	particles.push_back(particle3);

	//set camera 
	camera = new Camera();
	camera->setCameraLook(Vector3(0.0f, 0.0f, 0.0f));
	camera->setCameraPos(Vector3(0.0f, 0.0f,-750.0f));
	camera->setCameraUp(Vector3(0, 1, 0));
	
}

void Scene::render(float dt)
{


	// Clear Color and Depth Buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	gluLookAt(camera->getCameraPos().getX(), camera->getCameraPos().getY(), camera->getCameraPos().getZ(), camera->getCameraLook().getX(), camera->getCameraLook().getY(), camera->getCameraLook().getZ(), camera->getCameraUp().getX(), camera->getCameraUp().getY(), camera->getCameraUp().getZ());

	for (int i = 0; i < particles.size(); i++)
	{
		particles[i]->DrawParticle();
	}

	

	glutSwapBuffers();

}

void Scene::handleInput(float dt)
{
}

void Scene::update(float dt)
{
	angle += 0.5f * dt; 
	
	x = sinf(angle) * 2.0f;;
	y = cosf(angle) * 2.0f;;
	
	


	for (int i = 0; i < particles.size(); i++)
	{
		for (int j = 0; j < particles.size(); j++)
		{
			if (j != i)
			{
				Vector3 diff = particles[i]->position - particles[j]->position;
				float dist = diff.length();
				diff.normalise();
				particles[i]->velocity = particles[i]->velocity -   diff.dot((g *particles[j]->mass )/dist );
			}
		}
		particles[i]->Update();
	}

	



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
	farPlane = 1000.0f;

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