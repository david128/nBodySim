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
	particle = new Particle(50.0f, Vector3(0.0f, 2500.0f, 0.0f));
	particle->mass = 5.972e24f/d;
	particle2 = new Particle(250.0f, Vector3(0.0f, 0.0f, 0.0f));
	particle2->mass = 1.989e30f/d;
	particle3 = new Particle(50.0f, Vector3(0.0f, 2000.0f, 0.0f));
	particle3->mass = 5.972e24f / d;;
	particle->velocity = -200;
	particle3->velocity = -300;
	particle4 = new Particle(50.0f, Vector3(0.0f, 3000.0f, 0.0f));
	particle4->mass = 5.972e24f / d;;
	particle4->velocity = -120.0f;
	
	//store particles
	particles.push_back(particle);
	particles.push_back(particle2);
	particles.push_back(particle3);
	particles.push_back(particle4);

	InitCamera();
	
}

void Scene::InitCamera()
{
	//set camera 
	camera = new Camera();
	camera->setCameraLook(Vector3(0.0f, 0.0f, 0.0f));
	camera->setCameraPos(Vector3(0.0f, 0.0f, -9850.0f));
	camera->setCameraUp(Vector3(0, 1, 0));
	camera->SetDistanceToLook(9850.0);

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

	//update camera
	camera->update();

	
	if (time >= timeStep)
	{
		//update vel every time step
		//loop all particles
		for (int i = 0; i < particles.size(); i++)
		{
			//loop all particles 
			for (int j = 0; j < particles.size(); j++)
			{
				//if j and i are not same particle then calc j gravitational effect on i
				if (j != i)
				{
					//diff in positions
					Vector3 diff = particles[i]->position - particles[j]->position;
					float dist = diff.length(); //get distance
					
					float mult = (g * particles[j]->mass) / (dist * dist * dist); //multiplier  (g * mass )/ (distance ^3)

					Vector3 multDiff = Vector3(mult * diff.getX(), mult * diff.getY(), mult * diff.getZ()); //multiply  vector by multiplier to get force
					particles[i]->velocity = particles[i]->velocity - multDiff; //new V = old v + acceleration due to gravity

				}
			}
			particles[i]->Update(dt);//update particles with new forces
			
		}
		time = 0.0f;//reset time
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
	farPlane = 100000.0f;

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