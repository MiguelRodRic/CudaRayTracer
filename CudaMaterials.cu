#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "CudaMaterials.cuh"

__device__ bool LambertianScatter(const cudaRay& aRay, const cudaHitRecord& aHitRecord, curandState* aRandState, cuda3DVector& anAttenuation, cudaRay& aScattered)
{
	cuda3DVector scatterDirection = cuda3DVector{ aHitRecord.hitNormal } + CudaGetRandVectorInUnitSphere(aRandState);

	if (CudaCheckNearZero(scatterDirection))
		scatterDirection = aHitRecord.hitNormal; //avoid NaNs

	aScattered = cudaRay{ aHitRecord.point, scatterDirection };
	anAttenuation = aHitRecord.material.albedo;
	return true;
}


__device__ bool MetalScatter(const cudaRay& aRay, const cudaHitRecord& aHitRecord, curandState* aRandState, cuda3DVector& anAttenuation, cudaRay& aScattered)
{
	cuda3DVector reflected = CudaReflect(CudaNormalize(aRay.direction), aHitRecord.hitNormal);

	aScattered = cudaRay{ aHitRecord.point, reflected + (CudaGetRandVectorInUnitSphere(aRandState) * aHitRecord.material.fuzz) };
	anAttenuation = aHitRecord.material.albedo;

	return (CudaDotProduct(aScattered.direction, aHitRecord.hitNormal) > 0);
}


__device__ bool DielectricScatter(const cudaRay& aRay, const cudaHitRecord& aHitRecord, curandState* aRandState, cuda3DVector& anAttenuation, cudaRay& aScattered)
{
	anAttenuation = cuda3DVector{ 1.0f, 1.0f, 1.0f };
	float refractionRatio = (CudaDotProduct(aRay.direction, aHitRecord.hitNormal) < 0.0f) ? 1.0f / aHitRecord.material.aRefractionIndex : aHitRecord.material.aRefractionIndex;

	cuda3DVector direction = CudaNormalize(aRay.direction);

	float cosTheta = fmin(CudaDotProduct(-direction, aHitRecord.hitNormal), 1.0f);
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

	bool cannotRefract = refractionRatio * sinTheta > 1.0f;
	
	cuda3DVector bounced;
	if (cannotRefract) //Alternative: Use Schlick approximation || CudaGetReflectante(cosTheta, refractionRatio) > CudaGetRandInRange(aRandState, 0.0f, 1.0f))
		bounced = CudaReflect(direction, aHitRecord.hitNormal);
	else
		bounced = CudaRefract(direction, aHitRecord.hitNormal, refractionRatio);

	aScattered = cudaRay{ aHitRecord.point, bounced };
	return true;
}