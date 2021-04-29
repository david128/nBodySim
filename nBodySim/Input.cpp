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
