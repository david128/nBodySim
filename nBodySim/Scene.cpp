#include "Scene.h"


//temp




Scene::Scene(Input *inp)
{
	input = inp;
	
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


	//set up particles initial conditions
	//particle = new Particle(50.0f, Vector3(0.0f, 2500.0f, 0.0f));
	//particle->mass = 5.972e24f/d;
	//particle2 = new Particle(250.0f, Vector3(0.0f, 0.0f, 0.0f));
	//particle2->mass = 1.989e30f/d;
	//particle3 = new Particle(50.0f, Vector3(0.0f, 2000.0f, 0.0f));
	//particle3->mass = 5.972e24f / d;;
	//particle->velocity = -200;
	//particle3->velocity = -300;
	//particle4 = new Particle(50.0f, Vector3(0.0f, 3000.0f, 0.0f));
	//particle4->mass = 5.972e24f / d;;
	//particle4->velocity = -120.0f;
	//
	//store particles
	//particles.push_back(particle);
	//particles.push_back(particle2);
	//particles.push_back(particle3);
	//particles.push_back(particle4);

	particleManager = new ParticleManager(Vector3(10000.0f, 10000.0f, 10000.0f), g, 2);
	particleManager->InitTestSystem();

	Particle* p1 = new Particle(10, Vector3(0.0f, 0.0f, 0.0f), 60000000000000, Vector3(0.0f, 0.0f, 0.0f));
	Particle* p2 = new Particle(5, Vector3(0.0f, 35.0f, 0.0f), 50, Vector3(10.5f, 0.0f, 0.0f));

	//Particle* sun = new Particle(20.0f, Vector3(0.0f,0.0f,0.0f), 2e30, Vector3(0.0f,0.0f,0.0f));;
	//Particle* mercury = new Particle(20, Vector3(0.0f, 5.0e10f, 0.0f), 3.285e23, Vector3(47000.0f, 0.0f, 0.0f));
	//Particle* venus = new Particle(20, Vector3(0.0f, 1.1e11f, 0.0f), 4.8e24, Vector3(35000.0f, 0.0f, 0.0f));
	//Particle* earth = new Particle(20, Vector3(0.0f, 1.5e11f, 0.0f), 6e24, Vector3(30000.0f, 0.0f, 0.0f));
	//Particle* mars = new Particle(20, Vector3(0.0f, 2.2e11f, 0.0f), 2.4e24, Vector3(24000.0f, 0.0f, 0.0f));
	//Particle* jupiter = new Particle(20, Vector3(7.7e11f, 0.0f, 0.0f), 1e28, Vector3(0.0f, 13000.0f, 0.0f));
	//Particle* saturn = new Particle(20, Vector3(0.0f, 1.4e12f, 0.0f), 5.7e26, Vector3(9000.0f, 0.0f, 0.0f));
	//Particle* uranus = new Particle(20, Vector3(0.0f, 2.8e12f, 0.0f), 8.7e25, Vector3(6835.0f, 0.0f, 0.0f));
	//Particle* neptune = new Particle(20, Vector3(0.0f, 4.5e12f, 0.0f), 1e26, Vector3(5477.0f, 0.0f, 0.0f));
	//Particle* pluto = new Particle(20, Vector3(0.0f, 7.3e12f, 0.0f), 1.3e22, Vector3(4748.0f, 0.0f, 0.0f));


	//particleManager->AddParticle(sun);
	//particleManager->AddParticle(mercury);
	//particleManager->AddParticle(venus);
	//particleManager->AddParticle(earth);
	//particleManager->AddParticle(mars);
	//particleManager->AddParticle(jupiter);
	//particleManager->AddParticle(saturn);
	//particleManager->AddParticle(uranus);
	//particleManager->AddParticle(neptune);
	//particleManager->AddParticle(pluto);

	//particleManager->AddParticle(p1);
	//particleManager->AddParticle(p2);
	//particleManager->n = 2;

	InitCamera();
	
}

void Scene::InitCamera()
{
	//set camera 
	camera = new Camera();
	camera->setXzAngle(90.0f);
	camera->setCameraLook(Vector3(0.0f, 0.0f, 0.0f));
	camera->setCameraPos(Vector3(0.0f, 0.0f, 500.0f));
	camera->setCameraUp(Vector3(0, 1, 0));
	camera->SetDistanceToLook(500.0f);

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