#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "CudaHittable.cuh"

__device__ void CudaSetFaceNormal(const cudaRay& aRay, const cuda3DVector &anOutwardNormal, cuda3DVector &anOutNormal)
{
	bool frontFace = CudaDotProduct(aRay.direction, anOutwardNormal) < 0;

	anOutNormal = frontFace ? anOutwardNormal : cuda3DVector{ -anOutwardNormal.x, -anOutwardNormal.y, -anOutwardNormal.z };
}

__device__ bool CudaHitSphere(const cudaSphere& aSphere, float aTMin, float aTMax, const cudaRay& aRay, curandState* aRandState, cudaHitRecord& anOutHitRecord)
{
	cuda3DVector originToCenter = aRay.origin - aSphere.center;

	float a = CudaDotProduct(aRay.direction, aRay.direction);
	float halfB = CudaDotProduct(originToCenter, aRay.direction);
	float c = CudaDotProduct(originToCenter, originToCenter) - (aSphere.radius * aSphere.radius);

	float discriminant = halfB * halfB - (a * c);


	if (discriminant > 0)
	{
		float root = sqrt(discriminant);

		float posTemp = (-halfB + root) / a;
		float negTemp = (-halfB - root) / a;

		float temp = 0.0f;

		if (negTemp < aTMax && negTemp > aTMin)
			temp = negTemp;
		else if (posTemp < aTMax && posTemp > aTMin)
			temp = posTemp;

		if (temp != 0.0f)
		{
			anOutHitRecord.temp = temp;

			cuda3DVector rayAtTemp = aRay.origin + (aRay.direction * anOutHitRecord.temp);

			anOutHitRecord.point = rayAtTemp;

			cuda3DVector outwardNormal = (anOutHitRecord.point - aSphere.center) / aSphere.radius;

			anOutHitRecord.hitNormal = outwardNormal;

			anOutHitRecord.material = aSphere.material;

			CudaSetFaceNormal(aRay, outwardNormal, anOutHitRecord.hitNormal);
			return true;
		}
	}

	return false;
}

