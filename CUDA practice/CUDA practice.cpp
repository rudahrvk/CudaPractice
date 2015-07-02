// CUDA practice.cpp : Defines the entry point for the console application.
//
#pragma once

#include "stdafx.h"
#include "CUDA practice.h"

int _tmain(int argc, _TCHAR* argv[])
{
// 	Mat A, B, res;
// 	SIZE sizeA = {10000,10000};
// 	SIZE sizeB = {10000,10000};
// 	A.row = sizeA.cx; A.col = sizeA.cy; 
// 	B.row = sizeB.cx; B.col = sizeB.cy;
// 	A.elements = (float*)malloc(A.row*A.col*sizeof(float));
// 	B.elements = (float*)malloc(B.row*B.col*sizeof(float));
// 	res.elements = (float*)malloc(A.row*B.col*sizeof(float));
// 	for(int r=0; r<sizeA.cx; r++)
// 	{
// 		for(int c=0; c<sizeA.cy; c++)
// 		{
// 			*((float*)A.elements + A.col*r + c) = rand() + (rand()/(RAND_MAX)) ;
// 			*((float*)B.elements + B.col*c + r) = rand() + (rand()/(RAND_MAX)) ;
// 		}
// 	}
// 	
// 	LARGE_INTEGER t_start, t_end, t_freq;
// 	QueryPerformanceFrequency(&t_freq);
// 
// 	QueryPerformanceCounter(&t_start);
// 	MatMulCPU(A, B, res);
// 	QueryPerformanceCounter(&t_end);
// 	double cpuTime = (double)(t_end.QuadPart - t_start.QuadPart) / t_freq.QuadPart;
// 	printf("CPU time : %.6f\n", cpuTime);
// 
// 	QueryPerformanceCounter(&t_start);
// 	MatMulGPU(A, B, res);
// 	QueryPerformanceCounter(&t_end);
// 	double gpuTime = (double)(t_end.QuadPart - t_start.QuadPart) / t_freq.QuadPart;
// 	printf("GPU time : %.6f\n", gpuTime);
// 
// 	QueryPerformanceCounter(&t_start);
// 	MatMulGPU_Texture(A, B, res);
// 	QueryPerformanceCounter(&t_end);
// 	double textureTime = (double)(t_end.QuadPart - t_start.QuadPart) / t_freq.QuadPart;
// 	printf("GPU_texture time : %.6f\n", textureTime);

//	printf("CPU/GPU ratio : %.3f\nGPU/texGPU ratio : %.3f", (double)cpuTime/gpuTime, (double)gpuTime/textureTime);
//	printf("GPU/texture ratio : %.3f\n", (double)gpuTime/textureTime);

// 	free(A.elements);
// 	free(B.elements);

	const int inputSize = 1;
	const int winSq = 5 * 5;
	float* inputMat = (float*)malloc(inputSize*winSq*sizeof(float));
	for (int i = 0; i < inputSize*winSq; i++)
	{
		inputMat[i] = rand() / RAND_MAX;
	}

	int2* featurePos = (int2*)malloc(inputSize*sizeof(int2*));
	int* isGhost = (int*)malloc(inputSize*sizeof(int));
	float2* matchPoints = (float2*)malloc(inputSize*sizeof(float2*));
	for (int i = 0; i < inputSize; i++)
	{
		featurePos[i].x = featurePos[i].y = 0;
		isGhost[i] = false;
	}

	float* d_inputMat;
	cudaMalloc((void**)&d_inputMat, inputSize*winSq*sizeof(float));
	cudaMemcpy(d_inputMat, inputMat, inputSize*winSq*sizeof(float), cudaMemcpyHostToDevice);
	int2* d_featurePos;
	cudaMalloc((void**)&d_featurePos, inputSize*sizeof(int2*));
	cudaMemcpy(d_featurePos, featurePos, inputSize*sizeof(int2*), cudaMemcpyHostToDevice);
	int* d_isGhost;
	cudaMalloc((void**)&d_isGhost, inputSize*sizeof(int));
	cudaMemcpy(d_isGhost, isGhost, inputSize*sizeof(int), cudaMemcpyHostToDevice);
	float2* d_matchPoints;
	cudaMalloc((void**)&d_matchPoints, inputSize*sizeof(float2*));

	LARGE_INTEGER t_start, t_end, t_freq;
	QueryPerformanceFrequency(&t_freq);
	QueryPerformanceCounter(&t_start);
	ExtractFeaturePositions(d_inputMat, inputSize, d_featurePos, d_isGhost, d_matchPoints);
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&t_end);
	double gpuTime = (double)(t_end.QuadPart - t_start.QuadPart) / t_freq.QuadPart;
	printf("GPU time : %.6f\n", gpuTime);

	cudaMemcpy(matchPoints, d_matchPoints, inputSize*sizeof(float2*), cudaMemcpyDeviceToHost);

	std::cin.ignore();

	return 0;
}