#include "shape.h"
#include <math.h>

extern float verts[] = { -1.0, 1.0, 1.0,
-1.0, -1.0, 1.0,
1.0,  -1.0, 1.0,
1.0,  1.0, 1.0,			// front

-1.0, -1.0, -1.0,
-1.0,  -1.0, 1.0,
-1.0,  1.0, 1.0,
-1.0, 1.0, -1.0,		//Left

1.0, -1.0, -1.0,
-1.0,  -1.0, -1.0,
-1.0,  1.0, -1.0,
1.0, 1.0, -1.0,			// back

1.0, -1.0, 1.0,
1.0,  -1.0, -1.0,
1.0,  1.0, -1.0,
1.0, 1.0, 1.0,			//right

-1.0, 1.0, 1.0,
1.0,  1.0, 1.0,
1.0,  1.0, -1.0,
-1.0, 1.0, -1.0,		//top

-1.0, -1.0, 1.0,
-1.0, -1.0, -1.0,
1.0, -1.0, -1.0,
1.0, -1.0, 1.0
//bottom

};

extern float norms[] = { 0.0, 0.0, 1.0,		//0
0.0, 0.0, 1.0,		//1
0.0, 0.0, 1.0,		//2
0.0, 0.0, 1.0,		//3

-1.0, 0.0, 0.0,
-1.0, 0.0, 0.0,
-1.0, 0.0, 0.0,
-1.0, 0.0, 0.0,		//left

0.0, 0.0, -1.0,
0.0, 0.0, -1.0,
0.0, 0.0, -1.0,
0.0, 0.0, -1.0,		//back

1.0, 0.0, 0.0,
1.0, 0.0, 0.0,
1.0, 0.0, 0.0,
1.0, 0.0, 0.0,		//right	

0.0, 1.0, 0.0,
0.0, 1.0, 0.0,
0.0, 1.0, 0.0,
0.0, 1.0, 0.0,		//top	

0.0, -1.0, 0.0,
0.0, -1.0, 0.0,
0.0, -1.0, 0.0,
0.0, 1.0, 0.0		//bottom	

};


extern float texcoords[] = { 0.0, 0.0,
0.0, 1.0,
1.0, 1.0,
1.0, 0.0,
0.0, 0.0,
0.0, 1.0,
1.0, 1.0,
1.0, 0.0,

0.0, 0.0,
0.0, 1.0,
1.0, 1.0,
1.0, 0.0,

0.0, 0.0,
0.0, 1.0,
1.0, 1.0,
1.0, 0.0,

0.0, 0.0,
0.0, 1.0,
1.0, 1.0,
1.0, 0.0,

0.0, 0.0,
0.0, 1.0,
1.0, 1.0,
1.0, 0.0,
};

extern GLubyte indices[] = { 0, 1, 2, //front
0, 2, 3,
};

void Shape::render1()
{


	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, verts);
	glNormalPointer(GL_FLOAT, 0, norms);
	glTexCoordPointer(2, GL_FLOAT, 0, texcoords);

	glBegin(GL_QUADS);
	glArrayElement(0);
	glArrayElement(1);
	glArrayElement(2);
	glArrayElement(3);
	glEnd();

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);



}

void Shape::renderCube()
{

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, verts);
	glNormalPointer(GL_FLOAT, 0, norms);
	glTexCoordPointer(2, GL_FLOAT, 0, texcoords);


	glDrawArrays(GL_QUADS, 0, 24);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}


void Shape::renderDisc()
{
	float angle = (2 * 3.14159) / 20;
	float angle1;
	float angle2;
	float r = 1;


	for (int i = 0; i < 20; i++)
	{

		angle1 = i * angle;
		angle2 = (i + 1) * angle;

		glBegin(GL_TRIANGLES);

		glNormal3f(0.0f, 0.0f, 1.0f);
		glTexCoord2f(0.5f, 0.5f);
		glVertex3f(0.0f, 0.0f, 0.0f);

		glNormal3f(0.0f, 0.0f, 1.0f);
		glTexCoord2f(((cosf(angle1) / (r * 2)) + 0.5), ((sinf(angle1) / (r * 2)) + 0.5));
		glVertex3f((r * (cosf(angle1))), (r * (sinf(angle1))), 0.0);

		glNormal3f(0.0f, 0.0f, 1.0f);
		glTexCoord2f(((cosf(angle2) / (r * 2)) + 0.5), ((sinf(angle2) / (r * 2)) + 0.5));
		glVertex3f((r * (cosf(angle2))), (r * (sinf(angle2))), 0.0);


		glEnd();


	}


}

void Shape::renderCylDisc()
{
	float angle = (2 * 3.14159) / 20;
	float angle1;
	float angle2;
	float r = 1;


	for (int i = 0; i < 20; i++)
	{

		angle1 = i * angle;
		angle2 = (i + 1) * angle;

		glBegin(GL_TRIANGLES); //renders a disc of triangles

		glNormal3f(0.0f, 0.0f, -1.0f);
		glTexCoord2f(0.5f, 0.5f);
		glVertex3f(0.0f, 0.0f, 0.0f);

		glNormal3f(0.0f, 0.0f, -1.0f);
		glTexCoord2f(((cosf(angle1) / (r * 2)) + 0.5), ((sinf(angle1) / (r * 2)) + 0.25));
		glVertex3f((r * (cosf(angle1))), (r * (sinf(angle1))), 0.0);

		glNormal3f(0.0f, 0.0f, -1.0f);
		glTexCoord2f(((cosf(angle2) / (r * 2)) + 0.5), ((sinf(angle2) / (r * 2)) + 0.25));
		glVertex3f((r * (cosf(angle2))), (r * (sinf(angle2))), 0.0);


		glEnd();




		glBegin(GL_TRIANGLES); //renders a disc of triangles a unit above the current disc

		glNormal3f(0.0f, 0.0f, 1.0f);
		glTexCoord2f(0.5f, 0.5f);
		glVertex3f(0.0f, 0.0f, 1.0f);

		glNormal3f(0.0f, 0.0f, 1.0f);
		glTexCoord2f(((cosf(angle1) / (r * 2)) + 0.5), ((sinf(angle1) / (r * 2)) + 0.25));
		glVertex3f((r * (cosf(angle1))), (r * (sinf(angle1))), 1.0);

		glNormal3f(0.0f, 0.0f, 1.0f);
		glTexCoord2f(((cosf(angle2) / (r * 2)) + 0.5), ((sinf(angle2) / (r * 2)) + 0.25));
		glVertex3f((r * (cosf(angle2))), (r * (sinf(angle2))), 1.0);

		glEnd();



		glBegin(GL_QUADS); //renders quads between each triangle

		glNormal3f(0.0f, 0.0f, 1.0f);
		glTexCoord2f(((cosf(angle2) / (r * 2)) + 0.5), ((sinf(angle2) / (r * 2)) + 0.5));
		glVertex3f((r * (cosf(angle2))), (r * (sinf(angle2))), 1.0);

		glNormal3f(0.0f, 0.0f, 1.0f);
		glTexCoord2f(((cosf(angle1) / (r * 2)) + 0.5), ((sinf(angle1) / (r * 2)) + 0.5));
		glVertex3f((r * (cosf(angle1))), (r * (sinf(angle1))), 1.0);

		glNormal3f(0.0f, 0.0f, 1.0f);
		glTexCoord2f(((cosf(angle1) / (r * 2)) + 0.5), ((sinf(angle1) / (r * 2)) + 0.5));
		glVertex3f((r * (cosf(angle1))), (r * (sinf(angle1))), 0.0);

		glNormal3f(0.0f, 0.0f, 1.0f);
		glTexCoord2f(((cosf(angle2) / (r * 2)) + 0.5), ((sinf(angle2) / (r * 2)) + 0.5));
		glVertex3f((r * (cosf(angle2))), (r * (sinf(angle2))), 0.0);





		glEnd();

	}


}

void Shape::renderSphere()
{
	float angle = (2 * 3.14159) / 20;
	float delta = 3.14159 / 20;
	float angle1;
	float delta1;
	float r = 1;
	float u;
	float v;
	for (int i = 0; i < 20; i++)
	{

		for (int j = 0; j < 20; j++)
		{
			u = (1 / 20);
			v = (1 / 20);
			angle1 = angle * j;
			delta1 = delta * i;

			glBegin(GL_QUADS);

			glNormal3f(((r*(cosf(angle1))) *(sinf(delta1))), r*(cosf(delta1)), ((r*(sinf(angle1))) *(sinf(delta1))));
			glTexCoord2f(u*i, v*j);
			glVertex3f(((r*(cosf(angle1))) *(sinf(delta1))), r*(cosf(delta1)), ((r*(sinf(angle1))) *(sinf(delta1))));

			glNormal3f(((r*(cosf(angle1))) *(sinf(delta1 + delta))), r*(cosf(delta1 + delta)), ((r*(sinf(angle1))) *(sinf(delta1 + delta))));
			glTexCoord2f(u*i + 1, v*j);
			glVertex3f(((r*(cosf(angle1))) *(sinf(delta1 + delta))), r*(cosf(delta1 + delta)), ((r*(sinf(angle1))) *(sinf(delta1 + delta))));

			glNormal3f(((r*(cosf(angle1 + angle))) *(sinf(delta1 + delta))), r*(cosf(delta1 + delta)), ((r*(sinf(angle1 + angle))) *(sinf(delta1 + delta))));
			glTexCoord2f(u*i + 1, v*j + 1);
			glVertex3f(((r*(cosf(angle1 + angle))) *(sinf(delta1 + delta))), r*(cosf(delta1 + delta)), ((r*(sinf(angle1 + angle))) *(sinf(delta1 + delta))));

			glNormal3f(((r*(cosf(angle1 + angle))) *(sinf(delta1))), r*(cosf(delta1)), ((r*(sinf(angle1 + angle))) *(sinf(delta1))));
			glTexCoord2f(u*i, v*j + 1);
			glVertex3f(((r*(cosf(angle1 + angle))) *(sinf(delta1))), r*(cosf(delta1)), ((r*(sinf(angle1 + angle))) *(sinf(delta1))));

			glEnd();
		}
	}
}


