#include "CudaMaterials.cuh"
#include <curand_kernel.h>

typedef struct
{
	cuda3DVector origin;
	cuda3DVector horizontal;
	cuda3DVector vertical;
	cuda3DVector lowerLeftCorner;
	cuda3DVector u;
	cuda3DVector v;
	float lensRadius;
} cudaCamera;

__device__ cuda3DVector CudaGetRayColor(cudaRay& aRay, cudaHitRecord& aHitRecord, cudaSphere* aSomeSpheres, int aNumSpheres, curandState* aRandState, int& aDepth);

__global__ void CudaInitRandState(int aWidth, int aHeight, curandState* aRandStates);

__global__ void CudaGetColor(cudaCamera* aCamera, cudaSphere* aSomeSpheres, curandState* aRandState, cuda3DVector* aSomeInvValues, int aNumSpheres, int aWidth, int aHeight, int aResScale, int aNumSamplesPerPixel, cuda3DVector* anOutColor);

__global__ void CudaGetColorRecursive(cudaCamera* aCamera, cudaSphere* aSomeSpheres, curandState* aRandState, int aNumSpheres, int aWidth, int aHeight, int aResScale, cuda3DVector* anOutColor);

__global__ void CudaGetIdColor(cuda3DVector* anOutColor, int aNumElem);