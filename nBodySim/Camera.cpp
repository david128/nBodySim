#include "Camera.h"
#include <math.h>




void Camera::PanCamera(float angle)
{
	xzAngle += angle;
}

void Camera::ZoomCamera(float zoom)
{
	distanceToLook += zoom;
}


void Camera::setCameraPos(Vector3 i)
{
	position = i;
}


void Camera::setCameraLook(Vector3 k)
{
	look = k;
}

void Camera::setCameraUp(Vector3 j)
{
	up = j;
}

void Camera::SetDistanceToLook(float dist)
{
	distanceToLook = dist;
}

Vector3 Camera::getCameraPos()
{
	return position;
}

Vector3 Camera::getCameraLook()
{
	return look;
}

Vector3 Camera::getCameraUp()
{
	return up;
}

void Camera::setXzAngle(float xz)
{
	xzAngle =xz;
}



void Camera::update()
{
	//camera update not currently in use
	position.x = look.x + distanceToLook * cosf(xzAngle * degToRad);
	position.z = look.x + distanceToLook * sinf(xzAngle * degToRad);
	
}


