#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

__global__
void split(int* f)
{
	f = f + 1;
}