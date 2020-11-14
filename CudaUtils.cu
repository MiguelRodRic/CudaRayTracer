#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "device_launch_parameters.h"

#include "CudaUtils.cuh"

__global__ void CudaRand(unsigned int* anOutRandNum, int aNumElem)
{
	int i = ((blockDim.x * gridDim.x) * (blockDim.x * blockIdx.x + threadIdx.x)) + ((blockDim.y * blockIdx.y) + threadIdx.y);

	int next = (i) * 1103515245 + 12345;

	anOutRandNum[i] = (unsigned int)(next / 65536) % 32768;
	
}

__global__ void CudaInitCurand(curandState* aState, unsigned long aSeed)
{
	int index = ((blockDim.x * gridDim.x) * (blockDim.x * blockIdx.x + threadIdx.x)) + ((blockDim.y * blockIdx.y) + threadIdx.y);

	curand_init(aSeed, index, 0, &aState[index]);
}

__global__ void CudaInitCurandPerBlock(curandState* aState, unsigned long aSeed)
{
	int blockIndex = gridDim.x * blockIdx.x + blockIdx.y;

	curand_init(aSeed, blockIndex, 0, &aState[blockIndex]);
}

__device__ float CudaGetBlockRand(curandState* aState)
{
	int blockIndex = gridDim.x * blockIdx.x + blockIdx.y;
	return curand_uniform(&aState[blockIndex]);
}

__device__ float CudaGetRandInRange(curandState* aState, float aMin, float aMax)
{
	return aMin + (aMax - aMin) * curand_uniform(aState);
}

__device__ cuda3DVector CudaGetRandVectorInUnitSphere(curandState* aRandState)
{
	cuda3DVector randomVector;

	do
	{
		randomVector = (cuda3DVector{ curand_uniform(aRandState), curand_uniform(aRandState), curand_uniform(aRandState) } *2.0f) - cuda3DVector{ 1.0f, 1.0f, 1.0f };

	} while (CudaSquaredMagnitude(randomVector) >= 1.0f);

	return randomVector;
}

__device__ cuda3DVector CudaGetRandVectorInUnitDisk(curandState * aRandState)
{
	cuda3DVector randomVector;

	do
	{
		randomVector = (cuda3DVector{ curand_uniform(aRandState), curand_uniform(aRandState), 0.0f } *2.0f) - cuda3DVector{ 1.0f, 1.0f, 0.0f };
	} while (CudaSquaredMagnitude(randomVector) >= 1.0f);

	return randomVector;
}

__device__ cuda3DVector CudaNormalize(const cuda3DVector & aVector)
{
	float magnitude = sqrt(aVector.x * aVector.x + aVector.y * aVector.y + aVector.z * aVector.z);

	return aVector / magnitude;
}

__device__ cuda3DVector CudaReflect(const cuda3DVector & aVector, const cuda3DVector & aNormal)
{
	return aVector - (aNormal * CudaDotProduct(aVector, aNormal) * 2.0f);
}

__device__ cuda3DVector CudaRefract(const cuda3DVector & aVector, const cuda3DVector & aNormal, float aRefractionIndicesRatio)
{
	float cosTheta = fmin(CudaDotProduct(-aVector, aNormal), 1.0f);

	cuda3DVector perpendicularToNormalRay = (aVector + aNormal * cosTheta) * aRefractionIndicesRatio;
	cuda3DVector parallelToNormalRay = aNormal * -sqrt(fabs(1.0f - CudaSquaredMagnitude(perpendicularToNormalRay)));
	return parallelToNormalRay + perpendicularToNormalRay;
}

__device__ float CudaGetReflectante(float aCosine, float aRefractionRatio)
{
	//Schlick's approximation for reflectance
	float r0 = (1.0f - aRefractionRatio) / (1.0f + aRefractionRatio);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * pow((1.0f - aCosine), 5.0f);
}

__device__ bool CudaCheckNearZero(const cuda3DVector & aVector)
{
	const double s = 1e-8;
	return (fabs(aVector.x) < s) && (fabs(aVector.y) < s) && (fabs(aVector.z) < s);
}

__device__ float CudaSquaredMagnitude(const cuda3DVector & aVector)
{
	return aVector.x * aVector.x + aVector.y * aVector.y + aVector.z * aVector.z;
}

__device__ float CudaDotProduct(const cuda3DVector & aVector1, const cuda3DVector & aVector2)
{

	return (aVector1.x * aVector2.x + aVector1.y * aVector2.y + aVector1.z * aVector2.z);
}

__device__ float CudaDotProductNormalized(const cuda3DVector & aVector1, const cuda3DVector & aVector2)
{

	cuda3DVector normalizedVector1 = CudaNormalize(aVector1);
	cuda3DVector normalizedVector2 = CudaNormalize(aVector2);

	return (normalizedVector1.x * normalizedVector2.x + normalizedVector1.y * normalizedVector2.y + normalizedVector1.z * normalizedVector2.z);
}


__global__ void CudaDebugThreadID(unsigned int* anOutRandNum, int aNumElem)
{
	int i = (blockDim.x * threadIdx.x) + threadIdx.y;
	int j = (blockDim.y * threadIdx.y) + threadIdx.x;

	anOutRandNum[i] = (i + j) / 2;
}

__global__ void CudaDebugGlobalThreadID(unsigned int* anOutRandNum, int aNumElem)
{
	
	int i = ((blockDim.x * gridDim.x) * (blockDim.x * blockIdx.x + threadIdx.x)) + ((blockDim.y * blockIdx.y) + threadIdx.y);
	
	int j = ((blockDim.y * gridDim.y) * (blockDim.y * blockIdx.y + threadIdx.y)) + ((blockDim.x * blockIdx.x) + threadIdx.x);

	if (i < aNumElem && j < aNumElem)
		anOutRandNum[i] = (i + j) / 2;	
}


__global__ void CudaAddVectors(const float * A, const float * B, float * C, int numElements)
{
	int i = threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}