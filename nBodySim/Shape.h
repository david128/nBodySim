#ifndef _SHAPE_H
#define _SHAPE_H

#include <GL/freeglut.h>
#include <gl/gl.h>
#include <gl/glu.h>

class Shape
{

public:
	void render1();
	void renderCube();
	void renderDisc();
	void renderSphere();
	void renderCylDisc();
};
#endif 
