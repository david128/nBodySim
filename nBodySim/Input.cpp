#include "Input.h"

void Input::SetKeyDown(unsigned char key)
{
	keys[key].down = true;
}

void Input::SetKeyUp(unsigned char key)
{
	keys[key].down = false;
}

void Input::SetKeyPressed(unsigned char key)
{
	keys[key].pressed = true;
}

bool Input::isKeyDown(int key)
{
	return keys[key].down;
}

bool Input::isKeyPressed(int key)
{
	if (keys[key].down)
	{
		keys[key].down = false;
		return true;
	}
	else
	{
		return false;
	}
}

void Input::setMouseX(int x )
{
}

void Input::setMouseY(int y)
{
}

void Input::setMousePos(int x, int y)
{
}

int Input::getMouseX()
{
	return 0;
}

int Input::getMouseY()
{
	return 0;
}

void Input::setLeftMouseButton(bool b)
{
}

bool Input::isLeftMouseButtonPressed()
{
	return false;
}
