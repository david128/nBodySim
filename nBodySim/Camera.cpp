#include "Camera.h"
#include <math.h>


Camera::Camera()
{
}


Camera::~Camera()
{
}

void Camera::PanCamera(float angle)
{
	xzAngle += angle;
}

void Camera::ZoomCamera(float zoom)
{
	distanceToLook += zoom;
}

void Camera::RotateX(float rotate)
{
	
}

void Camera::RotateY(float rotate)
{

}

void Camera::RotateZ(float rotate)
{

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



void Camera::update()
{
	float cosR, cosP, cosY; //temp values for sin/cos from
	float sinR, sinP, sinY;

	//cosY = cosf(Yaw*3.1415 / 180);
	//cosP = cosf(Pitch*3.1415 / 180);
	//cosR = cosf(Roll*3.1415 / 180);
	//sinY = sinf(Yaw*3.1415 / 180);
	//sinP = sinf(Pitch*3.1415 / 180);
	//sinR = sinf(Roll*3.1415 / 180);

	position.x = look.x + distanceToLook * cosf(xzAngle);
	position.z = look.x + distanceToLook * sinf(xzAngle);
	
	/*forward.x = sinY * cosP;
	forward.y = sinP;
	forward.z = cosP * -cosY;
	up.x = -cosY * sinR - sinY * sinP * cosR;
	up.y = cosP * cosR;
	up.z = -sinY * sinR - sinP * cosR * -cosY;*/

	//right = forward.cross(up);

	//look = position;
	//look.add(forward);
}


