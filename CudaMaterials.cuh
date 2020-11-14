#include <curand_kernel.h>


#include "CudaHittable.cuh"

__device__ bool LambertianScatter(const cudaRay& aRay, const cudaHitRecord& aHitRecord, curandState* aRandState, cuda3DVector& anAttenuation, cudaRay& aScattered);
__device__ bool MetalScatter(const cudaRay& aRay, const cudaHitRecord& aHitRecord, curandState* aRandState, cuda3DVector& anAttenuation, cudaRay& aScattered);
__device__ bool DielectricScatter(const cudaRay& aRay, const cudaHitRecord& aHitRecord, curandState* aRandState, cuda3DVector& anAttenuation, cudaRay& aScattered);