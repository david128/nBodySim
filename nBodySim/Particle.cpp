#include "Particle.h"



Particle::Particle(float scale, Vector3 pos, float m) //alows initialisation of size and position
{
	size = scale;
	position = pos;
	mass = m;

}

Particle::Particle(float scale, Vector3 pos, float m, Vector3 v) //alows initialisation of size and position
{
	size = scale;
	position = pos;
	mass = m;
	velocity = v;

}

Particle::Particle()
{
}


void Particle::DrawParticle()
{

	glColor3f(colour.x,colour.y,colour.z);

	glPushMatrix();

	//translate to position
	glTranslatef(position.getX(), position.getY() , position.getZ() );
	//scale to size
	glScalef(size, size, size);
	   
	//renders sphere
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
	glPopMatrix();

	glColor3f(1.0f, 1.0, 1.0);
}

void Particle::SetColour(Vector3 col)
{
	colour = col;
}

