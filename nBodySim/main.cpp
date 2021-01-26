#include <GL/freeglut.h>
#include "Scene.h"
#include "Input.h"

void render(void);


Input* input;
Scene* scene;

int oldTimeSinceStart = 0;

void changeSize(int w, int h)
{
	scene->resize(w, h);
}

void handleKeyboard(unsigned char key, int x, int y)
{
	input->SetKeyDown(key);
}

void handleKeyboardUp(unsigned char key, int x, int y)
{
	input->SetKeyUp(key);
}


void render(void)
{
	int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
	float deltaTime = (float)timeSinceStart - (float)oldTimeSinceStart;
	oldTimeSinceStart = timeSinceStart;
	deltaTime = deltaTime / 100.0f;

	//clears buffer
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	scene->handleInput(deltaTime);
	scene->update(deltaTime);
	scene->render(deltaTime);
}


int	main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_STENCIL);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(800, 600);
	glutCreateWindow("N-Body dynamics");

	//input
	//keys
	glutKeyboardFunc(handleKeyboard);
	glutKeyboardUpFunc(handleKeyboardUp);

	//mouse
	/*glutMotionFunc();
	glutPassiveMotionFunc();
	glutMouseFunc();*/

	glutDisplayFunc(render);
	glutReshapeFunc(changeSize);
	glutIdleFunc(render);

	input = new Input();
	scene = new Scene(input);

	glutMainLoop();

	

}




