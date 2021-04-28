#pragma once
#include "Vector3.h"


class Camera
{
public:


	void PanCamera(float angle);
	void ZoomCamera(float zoom);

	void setCameraPos(Vector3);
	void setCameraLook(Vector3);
	void setCameraUp(Vector3);
	void SetDistanceToLook(float dist);

	Vector3 getCameraPos();
	Vector3 getCameraLook();
	Vector3 getCameraUp();
	void setXzAngle(float xz);

	void update();
	
private:
	Vector3 position = { 0, 0, 0 };
	Vector3 up = { 0, 1, 0 };
	Vector3 look;
	float distanceToLook;
	
	float xzAngle = 0;
	float degToRad = 3.1416 / 180.0f;
};




