#pragma once
class Input
{
	class Mouse
	{
		int x;
		int y;
		bool left;
		bool middle;
		bool right;
	};

	struct Key
	{
	public:
				
		bool down;
		bool pressed;

	};

public:

	
	void SetKeyDown(unsigned char key);
	void SetKeyUp(unsigned char key);
	void SetKeyPressed(unsigned char key);

	bool isKeyDown(int key);
	bool isKeyPressed(int key);

	void setMouseX(int x);
	void setMouseY(int y);
	void setMousePos(int x, int y);
	int getMouseX();
	int getMouseY();
	void setLeftMouseButton(bool b);
	bool isLeftMouseButtonPressed();

private:
	Key keys[255];
	Mouse mouse;
};

