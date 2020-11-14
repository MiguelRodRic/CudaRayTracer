#include "CudaMath.cuh"
#include <curand.h>
#include <curand_kernel.h>

__global__ void CudaRand(unsigned int* anOutRandNum, int aNumElem);

__global__ void CudaInitCurand(curandState* aState, unsigned long aSeed);

__global__ void CudaInitCurandPerBlock(curandState* aState, unsigned long aSeed);

__device__ float CudaGetBlockRand(curandState* aState);

__device__ float CudaGetRandInRange(curandState* aState, float aMin, float aMax);

__device__ cuda3DVector CudaGetRandVectorInUnitSphere(curandState* aRandState);

__device__ cuda3DVector CudaGetRandVectorInUnitDisk(curandState* aRandState);

__device__ cuda3DVector CudaNormalize(const cuda3DVector& aVector);

__device__ cuda3DVector CudaReflect(const cuda3DVector& aVector, const cuda3DVector& aNormal);

__device__ cuda3DVector CudaRefract(const cuda3DVector& aVector, const cuda3DVector& aNormal, float aRefractionIndicesRatio);

__device__ float CudaGetReflectante(float aCosine, float aRefractionRatio);

__device__ bool CudaCheckNearZero(const cuda3DVector& aVector);

__device__ float CudaSquaredMagnitude(const cuda3DVector& aVector);

__device__ float CudaDotProduct(const cuda3DVector& aVector1, const cuda3DVector& aVector2);

__device__ float CudaDotProductNormalized(const cuda3DVector& aVector1, const cuda3DVector& aVector2);

__global__ void CudaDebugThreadID(unsigned int* anOutRandNum, int aNumElem);

__global__ void CudaDebugGlobalThreadID(unsigned int* anOutRandNum, int aNumElem);

__global__ void CudaAddVectors(const float *A, const float *B, float *C, int numElements);
