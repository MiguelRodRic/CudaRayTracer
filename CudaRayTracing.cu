
#include "device_launch_parameters.h"

#include "CudaRayTracing.cuh"
#include <math_constants.h>

__device__ cuda3DVector CudaGetRayColor(cudaRay& aRay, cudaHitRecord& aHitRecord, cudaSphere* aSomeSpheres, int aNumSpheres, curandState* aRandState, int& aDepth)
{
	
	cudaRay& currentRay = aRay;
	cuda3DVector currentAttenuation{ 1.0f, 1.0f, 1.0f };
	for (int i = 0; i < aDepth; ++i)
	{
		float closestSoFar = 99999.0f;
		bool hitAnything = false;
		for (int i = 0; i < aNumSpheres; ++i)
		{
			if (CudaHitSphere(aSomeSpheres[i], 0.0001f, closestSoFar, currentRay, aRandState, aHitRecord))
			{
				hitAnything = true;
				closestSoFar = aHitRecord.temp;
			}
		}

		if (hitAnything)
		{
			//Diffuse without material scattering
			//cuda3DVector target = hitRecord.point + hitRecord.hitNormal + CudaNormalize(CudaGetRandVectorInUnitSphere(aRandState));
			//aRay = cudaRay{ hitRecord.point, target - hitRecord.point };
			//cuda3DVector color = CudaGetRayColor(aRay, hitRecord, aSpheres, aNumSpheres, aRandState, aDepth - 1);
			//return (color * 0.5f);

			//Material Scattering
			cudaRay scattered;
			cuda3DVector attenuation;
			bool hasScattered;

			switch (aHitRecord.material.type)
			{
			case LAMBERTIAN:
				hasScattered = LambertianScatter(aRay, aHitRecord, aRandState, attenuation, scattered);
				break;
			case METALLIC:
				hasScattered = MetalScatter(aRay, aHitRecord, aRandState, attenuation, scattered);
				break;
			case DIELECTRIC:
				hasScattered = DielectricScatter(aRay, aHitRecord, aRandState, attenuation, scattered);
				break;
			}

			if (hasScattered)
			{
				currentAttenuation = currentAttenuation * attenuation;
				currentRay = scattered;
			}
			else
			{
				return cuda3DVector{ 0.0f, 0.0f, 0.0f };;
			}
		}
		else
		{
			cuda3DVector normalizedDirection = CudaNormalize(aRay.direction);

			float t = 0.5f * (normalizedDirection.y + 1.0f);

			//linearly blend between blue and white
			cuda3DVector blueOperand{ t * 0.5f, t * 0.7f, t * 1.0f };
			cuda3DVector whiteOperand{ (1.0f - t) * 1.0f, (1.0f - t) * 1.0f, (1.0f - t) * 1.0f };

			cuda3DVector background = blueOperand + whiteOperand;

			return currentAttenuation * background;
		}
	}

	return cuda3DVector{ 0.0f, 0.0f, 0.0f };
}

__global__ void CudaInitRandState(int aWidth, int aHeight, curandState* aRandStates)
{
	const int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int j = (blockDim.y * blockIdx.y) + threadIdx.y;

	if (i >= aWidth || j >= aHeight)
		return;

	const int index = (aWidth * j) + i;
	
	curand_init(1994 + index, 0, 0, &aRandStates[index]);
}

__global__ void CudaGetColor(cudaCamera* aCamera, cudaSphere* aSomeSpheres, curandState* aRandState, cuda3DVector* aSomeInvValues, int aNumSpheres, int aWidth, int aHeight, int aResScale, int aNumSamplesPerPixel, cuda3DVector* anOutColor)
{
	int i = (blockDim.y * blockIdx.y) + threadIdx.y;
	int j = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (j > aSomeInvValues[1].y || i > aSomeInvValues[1].x) //if (j > (aHeight / aResScale) || i > (aWidth / aResScale))
		return;

	int pixelIndex;
	curandState localRandState;
	cuda3DVector color;
	for (int k = i * aResScale; k < (i * aResScale) + aResScale; ++k)
	{
		for (int l = j * aResScale; l < (j * aResScale) + aResScale; ++l)
		{
			if (l > aHeight || k > aWidth)
				return;
			
			pixelIndex = (aWidth * l) + k;
			localRandState = aRandState[(int)(pixelIndex * aSomeInvValues[0].z)]; /// (aResScale * aResScale)];
			color = cuda3DVector{ 0.0f, 0.0f, 0.0f };

			int maxDepth;

			for (int sample = 0; sample < aNumSamplesPerPixel; ++sample)
			{
				float u = float(k + curand_uniform(&localRandState)) * aSomeInvValues[0].x;// / (aWidth - 1);
				float v = float(l + curand_uniform(&localRandState)) * aSomeInvValues[0].y;// / (aHeight - 1);
				maxDepth = 10;
				cuda3DVector randomInDisk = CudaGetRandVectorInUnitDisk(&localRandState) * aCamera->lensRadius;
				cuda3DVector offset = aCamera->u * randomInDisk.x + aCamera->v * randomInDisk.y;
				cuda3DVector rayDirection = aCamera->lowerLeftCorner + (aCamera->horizontal * u) + (aCamera->vertical * v) - aCamera->origin - offset;

				cudaRay ray = { aCamera->origin + offset, rayDirection };
				cudaHitRecord sharedRecord;

				cuda3DVector sampleColor = CudaGetRayColor(ray, sharedRecord, aSomeSpheres, aNumSpheres, &localRandState, maxDepth);
				color = color + sampleColor;
			}
			anOutColor[pixelIndex] = color; 
		}
	}
}

__global__ void CudaGetColorRecursive(cudaCamera* aCamera, cudaSphere* aSomeSpheres, curandState* aRandState, int aNumSpheres, int aWidth, int aHeight, int aNumSamplesPerPixel, cuda3DVector* anOutColor)
{
	int i = (blockDim.y * blockIdx.y) + threadIdx.y;
	int j = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (j > aHeight || i > aWidth )
		return;
	
	const int index = (aWidth * j) + i;
	curandState localRandState = aRandState[index];

	cuda3DVector color{ 0.0f, 0.0f, 0.0f };
	for (int sample = 0; sample < aNumSamplesPerPixel; ++sample)
	{
		float u = float(i + curand_uniform(&localRandState)) / (aWidth - 1);
		float v = float(j + curand_uniform(&localRandState)) / (aHeight - 1);

		int maxDepth = 10;
		cuda3DVector randomInDisk = CudaGetRandVectorInUnitDisk(&localRandState) * aCamera->lensRadius;
		cuda3DVector offset = aCamera->u * randomInDisk.x + aCamera->v * randomInDisk.y;
		cuda3DVector rayDirection = aCamera->lowerLeftCorner + (aCamera->horizontal * u) + (aCamera->vertical * v) - aCamera->origin - offset;

		cudaRay ray = { aCamera->origin + offset, rayDirection };
		cudaHitRecord sharedRecord;

		cuda3DVector sampleColor = CudaGetRayColor(ray, sharedRecord, aSomeSpheres, aNumSpheres, &localRandState, maxDepth);
		color = color + sampleColor;
	}
	anOutColor[index] = color; 
}

__global__ void CudaGetIdColor(cuda3DVector* anOutColor, int aNumElem)
{
	int index = ((blockDim.x * gridDim.x) * (blockDim.x * blockIdx.x + threadIdx.x)) + ((blockDim.y * blockIdx.y) + threadIdx.y);

	int indexY = ((blockDim.y * gridDim.y) * (blockDim.y * blockIdx.y + threadIdx.y)) + ((blockDim.x * blockIdx.x) + threadIdx.x);

	anOutColor[index].x = (float)index;
	anOutColor[index].y = (float)indexY;
	anOutColor[index].z = index + indexY / 2;
}