#include "Camera.h"
#include <math.h>


Camera::Camera()
{
}


Camera::~Camera()
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

void Camera::setForward(Vector3 o)
{
	forward = o;
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

Vector3 Camera::getForward()
{
	return forward;
}

void Camera::update()
{
	float cosR, cosP, cosY; //temp values for sin/cos from
	float sinR, sinP, sinY;

	cosY = cosf(Yaw*3.1415 / 180);
	cosP = cosf(Pitch*3.1415 / 180);
	cosR = cosf(Roll*3.1415 / 180);
	sinY = sinf(Yaw*3.1415 / 180);
	sinP = sinf(Pitch*3.1415 / 180);
	sinR = sinf(Roll*3.1415 / 180);

	forward.x = sinY * cosP;
	forward.y = sinP;
	forward.z = cosP * -cosY;
	up.x = -cosY * sinR - sinY * sinP * cosR;
	up.y = cosP * cosR;
	up.z = -sinY * sinR - sinP * cosR * -cosY;

	right = forward.cross(up);

	look = position;
	look.add(forward);
}

void Camera::moveForward(float dt)
{
	forward.scale(moveSpeed *dt);
	position.add(forward);
}

void Camera::moveBackward(float dt)
{
	forward.scale(moveSpeed *dt);
	position.subtract(forward);
}

void Camera::moveRight(float dt)
{
	right.scale(moveSpeed *dt);
	position.add(right);
}

void Camera::moveLeft(float dt)
{
	right.scale(moveSpeed *dt);
	position.subtract(right);
}

void Camera::rotateUp(float dt)
{
	Pitch += lookSpeed * dt;
}

void Camera::rotateDown(float dt)
{
	Pitch -= lookSpeed * dt;
}

void Camera::rotateRight(float dt)
{
	Yaw += lookSpeed * dt;
}

void Camera::rotateLeft(float dt)
{
	Yaw -= lookSpeed * dt;
}

void Camera::restart()
{
	position = { 0, 0, 6 };
	up = { 0, 1, 0 };
}

void Camera::Mouse(float dt, int w, int h, int X, int Y)
{
	//Sets mouse coords
	mouseX = X;
	mouseY = Y;

	//Limits camera movement
	if ((mouseY - (h / 2)) > 0)
		Pitch -= lookSpeed * dt;
	if ((mouseY - (h / 2)) < 0)
		Pitch += lookSpeed * dt;

	//Mouse moves to the right
	if ((mouseX - (w / 2)) > 0)
		Yaw += lookSpeed * dt;

	//Mouse moves to the left
	else if ((mouseX - (w / 2)) < 0)
		Yaw -= lookSpeed * dt;

	update();
}