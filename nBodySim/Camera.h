#pragma once
#include "Vector3.h"


class Camera
{
public:
	Camera();
	~Camera();



	void setCameraPos(Vector3);
	void setCameraLook(Vector3);
	void setCameraUp(Vector3);
	void setForward(Vector3);

	Vector3 getCameraPos();
	Vector3 getCameraLook();
	Vector3 getCameraUp();
	Vector3 getForward();

	void update();
	void moveForward(float);
	void moveBackward(float);
	void moveRight(float);
	void moveLeft(float);
	void restart();
	void Mouse(float, int, int, int, int);
	void rotateUp(float);
	void rotateDown(float);
	void rotateRight(float);
	void rotateLeft(float);
private:
	Vector3 position = { 0, 0, 6 };
	Vector3 up = { 0, 1, 0 };

	Vector3 forward, look, right;

	int moveSpeed = 1, lookSpeed = 10;

	float Yaw = 0.0f;
	float Pitch = 0.0f;
	float Roll = 0.0f;
	float mouseX, mouseY;
};




