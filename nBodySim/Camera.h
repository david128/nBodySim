#pragma once
#include "Vector3.h"


class Camera
{
public:
	Camera();
	~Camera();

	void PanCamera(float angle);
	void ZoomCamera(float zoom);
	void RotateX(float rotate);
	void RotateY(float rotate);
	void RotateZ(float rotate);

	void setCameraPos(Vector3);
	void setCameraLook(Vector3);
	void setCameraUp(Vector3);
	void SetDistanceToLook(float dist);

	Vector3 getCameraPos();
	Vector3 getCameraLook();
	Vector3 getCameraUp();
	Vector3 getForward();

	void update();
	
private:
	Vector3 position = { 0, 0, 0 };
	Vector3 up = { 0, 1, 0 };
	Vector3 look;
	float distanceToLook;
	
	float xzAngle = 0;

	   	 

	float Yaw = 0.0f;
	float Pitch = 0.0f;
	float Roll = 0.0f;
	float mouseX, mouseY;
};




