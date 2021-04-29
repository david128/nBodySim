#pragma once
class Input
{
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

private:
	Key keys[255];
};

