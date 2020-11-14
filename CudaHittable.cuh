#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "CudaUtils.cuh"

enum MaterialType
{
	LAMBERTIAN,
	METALLIC,
	DIELECTRIC
};

typedef struct
{
	cuda3DVector origin;
	cuda3DVector direction;
} cudaRay;

typedef struct
{
	MaterialType type;
	cuda3DVector albedo;
	float fuzz;
	float aRefractionIndex;
} cudaMaterial;

typedef struct
{
	float temp;
	cuda3DVector point;
	cuda3DVector hitNormal;
	cudaMaterial material;
} cudaHitRecord;

typedef struct
{
	cuda3DVector center;
	float radius;
	cudaMaterial material;
} cudaSphere;

__device__ void CudaSetFaceNormal(const cudaRay& aRay, const cuda3DVector& anOutwardNormal, cuda3DVector& anOutNormal);

__device__ bool CudaHitSphere(const cudaSphere& aSphere, float aTMin, float aTMax, const cudaRay& aRay, curandState* aRandState, cudaHitRecord& anOutHitRecord);

