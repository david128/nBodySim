#include <GL/freeglut.h>
#include "Scene.h"

void render(void);

Scene* scene = new Scene();

int oldTimeSinceStart = 0;

void changeSize(int w, int h)
{
	scene->resize(w, h);
}

int	main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_STENCIL);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(800, 600);
	glutCreateWindow("N-Body dynamics");

	glutDisplayFunc(render);
	glutReshapeFunc(changeSize);
	glutIdleFunc(render);

	glutMainLoop();

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