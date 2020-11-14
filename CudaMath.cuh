typedef struct
{
	float x, y, z;
} cuda3DVector;

__device__ cuda3DVector operator-(cuda3DVector aVector);

__device__ cuda3DVector operator+(cuda3DVector aVector1, cuda3DVector aVector2);

__device__ cuda3DVector operator-(cuda3DVector aVector1, cuda3DVector aVector2);

__device__ cuda3DVector operator*(cuda3DVector aVector1, cuda3DVector aVector2);

__device__ cuda3DVector operator+(cuda3DVector aVector, float aNumber);

__device__ cuda3DVector operator-(cuda3DVector aVector, float aNumber);

__device__ cuda3DVector operator*(cuda3DVector aVector, float aNumber);

__device__ cuda3DVector operator/(cuda3DVector aVector, float aNumber);

__device__ cuda3DVector operator-(float aNumber, cuda3DVector aVector);
