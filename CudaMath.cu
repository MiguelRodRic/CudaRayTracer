#include <curand.h>
#include <curand_kernel.h>
#include "CudaMath.cuh"

__device__ cuda3DVector operator+(cuda3DVector aVector1, cuda3DVector aVector2)
{
	return cuda3DVector{ aVector1.x + aVector2.x, aVector1.y + aVector2.y, aVector1.z + aVector2.z };
}

__device__ cuda3DVector operator-(cuda3DVector aVector1, cuda3DVector aVector2)
{
	return cuda3DVector{ aVector1.x - aVector2.x, aVector1.y - aVector2.y, aVector1.z - aVector2.z };
}

__device__ cuda3DVector operator*(cuda3DVector aVector1, cuda3DVector aVector2)
{
	return cuda3DVector{ aVector1.x * aVector2.x, aVector1.y * aVector2.y, aVector1.z * aVector2.z };
}

__device__ cuda3DVector operator+(cuda3DVector aVector, float aNumber)
{
	return cuda3DVector{ aVector.x + aNumber, aVector.y + aNumber, aVector.z + aNumber };
}

__device__ cuda3DVector operator-(cuda3DVector aVector, float aNumber)
{
	return cuda3DVector{ aVector.x - aNumber, aVector.y - aNumber, aVector.z - aNumber };
}

__device__ cuda3DVector operator*(cuda3DVector aVector, float aNumber)
{
	return cuda3DVector{ aVector.x * aNumber, aVector.y * aNumber, aVector.z * aNumber };
}

__device__ cuda3DVector operator/(cuda3DVector aVector, float aNumber)
{
	return cuda3DVector{ aVector.x / aNumber, aVector.y / aNumber, aVector.z / aNumber };
}