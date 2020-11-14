
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// kernel.cu: Defines the entry point for the console application.
//

#include <fstream>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>
#include <thread>
#include <algorithm>

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "CudaRayTracing.cuh"

#define PIXELWIDTH 600
#define PIXELHEIGHT 360
#define M_PI 3.14159265358979323846f


using namespace std;

float clamp(float value, float min, float max)
{
	if (value < min)
		return min;
	if (value > max)
		return max;

	return value;
}

#pragma optimize ("", off)
void PrintImage(cuda3DVector* aPixelColors, int aNSamples)
{
	int nx = PIXELWIDTH, ny = PIXELHEIGHT;
	ofstream outfile;
	outfile.open("sample.ppm");
	if (outfile.is_open())
	{
		cout << "Writing into the file... \n" << endl;
		outfile << "P3\n" << nx << " " << ny << "\n255\n";

		
		for (int j = ny - 1; j >= 0; --j)
		{

			for (int i = 0; i < nx; ++i)
			{
				int ir, ig, ib;
				ir = int(aPixelColors[i + PIXELWIDTH * j].x);
				ig = int(aPixelColors[i + PIXELWIDTH * j].y);
				ib = int(aPixelColors[i + PIXELWIDTH * j].z);
				//gamma correction
				float scale = 1.0f / aNSamples;
				ir = 256.0f * clamp(sqrt(scale * ir), 0.0f, 0.9999f);
				ig = 256.0f * clamp(sqrt(scale * ig), 0.0f, 0.9999f);
				ib = 256.0f * clamp(sqrt(scale * ib), 0.0f, 0.9999f);

				outfile << ir << " " << ig << " " << ib << "\n";
			}
		}

		outfile.close();
	}
}


//Some random helpers
float GetRandomInRange(float aMin, float aMax)
{
	float random = (float)rand() / (float)RAND_MAX;
	float range = aMax - aMin;
	return random * range + aMin;
}

float GetMagnitude(cuda3DVector aVector)
{
	return sqrt(aVector.x * aVector.x + aVector.y * aVector.y + aVector.z * aVector.z);
}

cuda3DVector Normalize(cuda3DVector aVector)
{
	float magnitude = GetMagnitude(aVector);

	return cuda3DVector{ aVector.x / magnitude, aVector.y / magnitude, aVector.z / magnitude };
}


cuda3DVector CrossProduct(cuda3DVector aVector1, cuda3DVector aVector2)
{
	return cuda3DVector
	{
		(aVector1.y * aVector2.z) - (aVector1.z * aVector2.y),
		(aVector1.z * aVector2.x) - (aVector1.x * aVector2.z),
		(aVector1.x * aVector2.y) - (aVector1.y * aVector2.x)
	};
}

int main()
{
	//Seeding the random generation 
	srand(static_cast <unsigned> (time(0)));

	auto start = chrono::system_clock::now();


	//wait for GPU to finish
	cudaError err = cudaSuccess;

	int numElem = PIXELWIDTH * PIXELHEIGHT;
	int resolutionScale = 1; //Pixels per axis per thread
	int numSpp = 30;
	int numSpheres = 200;
	//memory buffers sizes
	size_t colorSize = numElem * sizeof(cuda3DVector);
	size_t sphereSize = numSpheres * sizeof(cudaSphere);
	size_t randStateSize = numElem / (resolutionScale * resolutionScale) * sizeof(curandState);

	//Allocating CPU arrays
	cuda3DVector* hostColorPtr = (cuda3DVector*)malloc(colorSize);
	cudaSphere* hostSpherePtr = (cudaSphere*)malloc(sphereSize);
	cudaCamera* hostCameraPtr = (cudaCamera*)malloc(sizeof(cudaCamera));

	if (hostColorPtr == NULL ||
		hostSpherePtr == NULL ||
		hostCameraPtr == NULL)
	{
		fprintf(stderr, "Failed to allocate host elements!\n");
		exit(EXIT_FAILURE);
	}

	//Initializing CPU data
	for (int i = 0; i < numElem; ++i)
	{
		hostColorPtr[i].x = 0.0f;
		hostColorPtr[i].y = 0.0f;
		hostColorPtr[i].z = 0.0f;
	}

	float aspectRatio = 16.0f / 9.0f;
	float verticalFOV = 20.0f;
	float aperture = 0.1f;
	cuda3DVector lookFrom{ 13.0f, 2.0f, 3.0f };
	cuda3DVector lookAt{ 0.0f, 0.0f, 0.0f };
	cuda3DVector vUp{ 0.0f, 1.0f, 0.0f };

	float focusDistance = 10.0f; //Alternative: GetMagnitude(cuda3DVector{ lookFrom.x - lookAt.x, lookFrom.y - lookAt.y, lookFrom.z - lookAt.z });
	
	float theta = verticalFOV * M_PI / 180.0f; //Degrees to Radians
	float h = tan(theta / 2.0f);

	float viewportHeight = 2.0f * h;
	float viewportWidth = aspectRatio * viewportHeight;

	cuda3DVector w = Normalize(cuda3DVector{ lookFrom.x - lookAt.x,  lookFrom.y - lookAt.y, lookFrom.z - lookAt.z });
	cuda3DVector u = Normalize(CrossProduct(vUp, w));
	cuda3DVector v = CrossProduct(w, u);

	cuda3DVector origin = lookFrom;
	cuda3DVector horizontal = cuda3DVector{ focusDistance * u.x * viewportWidth, focusDistance * u.y * viewportWidth, focusDistance * u.z * viewportWidth };
	cuda3DVector vertical = cuda3DVector{ focusDistance * v.x * viewportHeight, focusDistance * v.y * viewportHeight, focusDistance * v.z * viewportHeight };

	cuda3DVector lowerLeftCorner
	{
		origin.x - horizontal.x / 2.0f - vertical.x / 2.0f - focusDistance * w.x,
		origin.y - horizontal.y / 2.0f - vertical.y / 2.0f - focusDistance * w.y,
		origin.z - horizontal.z / 2.0f - vertical.z / 2.0f - focusDistance * w.z
	};

	float lensRadius = aperture / 2.0f;
	
	*hostCameraPtr = {origin, horizontal, vertical, lowerLeftCorner, u, v, lensRadius};
	
	hostSpherePtr[0] = cudaSphere{ cuda3DVector{-4.0f, 1.0f, 0.0f}, 1.0f, cudaMaterial{ LAMBERTIAN, cuda3DVector{0.4f, 0.2f, 0.1f}, 0.5f, 1.5f} };
	hostSpherePtr[1] = cudaSphere{ cuda3DVector{0.0f, -1000.0f, 0.0f}, 1000.0f, cudaMaterial{ LAMBERTIAN, cuda3DVector{0.5f, 0.5f, 0.5f}, 0.0f, 1.5f} };
	hostSpherePtr[2] = cudaSphere{ cuda3DVector{0.0f, 1.0f, 0.0f}, 1.0f, cudaMaterial{ DIELECTRIC, cuda3DVector{1.0f, 1.0f, 1.0f}, 0.1f, 4.0f} };
	hostSpherePtr[3] = cudaSphere{ cuda3DVector{4.0f, 1.0f, 0.0f}, 1.0f, cudaMaterial{ METALLIC, cuda3DVector{0.7f, 0.6f, 0.5f}, 0.0f, 1.5f} };
	int index = 0;
	for (int i = -10; i < 10; ++i)
	{
		for (int j = -5; j < 5; ++j, ++index)
		{
			
			if (index > 3 && index < numSpheres)
			{
				float material = GetRandomInRange(0, 3);

				cuda3DVector center{i + 0.9f * GetRandomInRange(-1.0f, 1.0f), 0.2f, j + 0.9f * GetRandomInRange(-1.0f, 1.0f) };

				cuda3DVector boundaryCheck{ center.x - 4.0f, center.y - 0.2f, center.z - 0.0f };
				if (GetMagnitude(boundaryCheck) > 0.9f)
				{
					if (material < 1.5f) //diffuse
					{
						cuda3DVector albedo{ GetRandomInRange(0.0f, 1.0f) * GetRandomInRange(0.0f, 1.0f), GetRandomInRange(0.0f, 1.0f) * GetRandomInRange(0.0f, 1.0f), GetRandomInRange(0.0f, 1.0f) * GetRandomInRange(0.0f, 1.0f) };
						hostSpherePtr[index] = cudaSphere{ center, 0.2f, cudaMaterial{ LAMBERTIAN, albedo, 0.0f, 0.0f} };
					}
					else if (material < 2.5f) //metallic
					{
						cuda3DVector albedo{ GetRandomInRange(0.5f, 1.0f), GetRandomInRange(0.5f, 1.0f), GetRandomInRange(0.5f, 1.0f) };
						float fuzz = GetRandomInRange(0.0f, 0.5f);
						hostSpherePtr[index] = cudaSphere{ center, 0.2f, cudaMaterial{ METALLIC, albedo, fuzz, 0.0f} };
					}
					else // dielectric
					{
						cuda3DVector albedo{ 1.0f, 1.0f, 1.0f };
						hostSpherePtr[index] = cudaSphere{ center, 0.2f, cudaMaterial{ DIELECTRIC, albedo, 0.0f, 5.0f} };
					}
				}
			}
		}

	}

	//avoid as many GPU divisions as posible
	cuda3DVector* hostInvValues = (cuda3DVector*)malloc(sizeof(cuda3DVector) * 2);
	hostInvValues[0].x = (1.0f / (PIXELWIDTH - 1));
	hostInvValues[0].y = (1.0f / (PIXELHEIGHT - 1));
	hostInvValues[0].z = (1.0f / (resolutionScale * resolutionScale));
	hostInvValues[1].x = PIXELWIDTH / resolutionScale;
	hostInvValues[1].y = PIXELHEIGHT / resolutionScale;

	//Allocating GPU data
	cuda3DVector* deviceColorPtr = NULL;
	checkCudaErrors(cudaMalloc((void**)&deviceColorPtr, colorSize));

	cudaSphere* deviceSpherePtr = NULL;
	checkCudaErrors(cudaMalloc((void**)&deviceSpherePtr, sphereSize));
	cudaCamera* deviceCameraPtr = NULL;
	checkCudaErrors(cudaMalloc((void**)&deviceCameraPtr, sizeof(cudaCamera)));

	curandState* deviceRandStatePtr = NULL;
	checkCudaErrors(cudaMalloc((void**)&deviceRandStatePtr, randStateSize));

	cuda3DVector* deviceInvValues = NULL;
	checkCudaErrors(cudaMalloc((void**)&deviceInvValues, sizeof(cuda3DVector) * 2));

	checkCudaErrors(cudaDeviceSynchronize());

	//Copying CPU to GPU data
	
	checkCudaErrors(cudaMemcpy(deviceColorPtr, hostColorPtr, colorSize, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(deviceSpherePtr, hostSpherePtr, sphereSize, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(deviceCameraPtr, hostCameraPtr, sizeof(cudaCamera), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMemcpy(deviceInvValues, hostInvValues, sizeof(cuda3DVector) * 2, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());
	
	//executing kernels
	auto endInit = chrono::system_clock::now();

	cout << "Ellapsed Time for initialization: " << (float)(chrono::duration_cast<std::chrono::microseconds>(endInit - start).count() / 1000.0f) << " milliseconds" << endl;


	dim3 threadsPerBlock(8, 8, 1);
	
	dim3 blocksPerGrid;

	blocksPerGrid.x = (max(PIXELWIDTH / resolutionScale, PIXELHEIGHT / resolutionScale) + (threadsPerBlock.x) - 1) / (threadsPerBlock.x);
	blocksPerGrid.y = (max(PIXELWIDTH / resolutionScale, PIXELHEIGHT / resolutionScale) + (threadsPerBlock.y) - 1) / (threadsPerBlock.y);
	blocksPerGrid.z = 1;

	dim3 blocks(PIXELWIDTH / resolutionScale / threadsPerBlock.x + 1, PIXELHEIGHT / resolutionScale / threadsPerBlock.y + 1);
	CudaInitRandState << <blocks, threadsPerBlock >> > (PIXELWIDTH / resolutionScale, PIXELHEIGHT / resolutionScale, deviceRandStatePtr);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	auto endRand = chrono::system_clock::now();


	cout << "Ellapsed Time for rand state generation: " << (float)(chrono::duration_cast<std::chrono::microseconds>(endRand - endInit).count() / 1000.0f) << " milliseconds" << endl;
	
	CudaGetColor<< < blocksPerGrid, threadsPerBlock >> > (deviceCameraPtr, deviceSpherePtr, deviceRandStatePtr, deviceInvValues, numSpheres, PIXELWIDTH, PIXELHEIGHT, resolutionScale, numSpp, deviceColorPtr);
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	auto endRT = chrono::system_clock::now();


	cout << "Ellapsed Time for Ray Tracing: " << (float) (chrono::duration_cast<std::chrono::microseconds>(endRT - endRand).count() / 1000.0f) << " milliseconds" << endl;

	//Copy back from GPU to CPU
	checkCudaErrors(cudaMemcpy(hostColorPtr, deviceColorPtr, colorSize, cudaMemcpyDeviceToHost));
		   
	//releasing GPU memory
	checkCudaErrors(cudaFree(deviceCameraPtr)); 

	checkCudaErrors(cudaFree(deviceRandStatePtr));

	checkCudaErrors(cudaFree(deviceSpherePtr));
	
	checkCudaErrors(cudaFree(deviceColorPtr));

	checkCudaErrors(cudaFree(deviceInvValues));
	
	auto end1 = chrono::system_clock::now();

	PrintImage(hostColorPtr, numSpp);

	auto end2 = chrono::system_clock::now();

	cout << "Ellapsed Time for printing: " << chrono::duration_cast<std::chrono::milliseconds>(end2 - end1).count() << " milliseconds" << endl;

	//releasing CPU memory
	free(hostCameraPtr);
	free(hostSpherePtr);
	free(hostColorPtr);

	cout << "Press any key to exit" << endl;
	getchar();
	return 0;
}


