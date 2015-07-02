#pragma once

#include "cuda_runtime.h"
#include "stdafx.h"

#define FEATURE_WIDTH				32
#define FEATURE_HEIGHT				32

#define SEARCH_WIDTH		(FEATURE_WIDTH * 2 - 1)
#define SEARCH_HEIGHT		(FEATURE_HEIGHT * 2 - 1)
#define RESULT_WIDTH		(SEARCH_WIDTH - FEATURE_WIDTH + 1)		// FEATURE_WIDTH
#define RESULT_HEIGHT		(SEARCH_HEIGHT - FEATURE_HEIGHT + 1)		// FEATURE_HEIGHT

#define WINDOW_SIZE					5

__global__ void ExtractFeaturePositionsKernel(float* d_results, int featureSize, int2* d_featurePositions, int* d_ghostFeature, float2* d_matchPoints)
{
	__shared__ float s_maxResult[RESULT_WIDTH];
	__shared__ int s_maxIndexX[RESULT_WIDTH];
	__shared__ int s_maxIndexY[RESULT_WIDTH];

	if (d_ghostFeature[blockIdx.x] == 0)		//invalid Feature point exception
		return;

	int myBlockIdx = blockIdx.x * RESULT_WIDTH * RESULT_HEIGHT;
	d_results += myBlockIdx;
	d_featurePositions += myBlockIdx;
	d_matchPoints += myBlockIdx;

	s_maxResult[threadIdx.x] = d_results[threadIdx.x];
	s_maxIndexX[threadIdx.x] = threadIdx.x;
	s_maxIndexY[threadIdx.x] = 0;

	//calculate each column max
	for (int row = 1; row < RESULT_HEIGHT; ++row)
	{
		if (s_maxResult[threadIdx.x] < d_results[threadIdx.x + row * RESULT_WIDTH])
		{
			s_maxResult[threadIdx.x] = d_results[threadIdx.x + row * RESULT_WIDTH];
			s_maxIndexY[threadIdx.x] = row;
		}
	}
	__syncthreads();

	//calculate max result
	for (int i = 2; i <= blockDim.x; i <<= 1)
	{
		int dist = blockDim.x / i;
		int compIdx = threadIdx.x + dist;
		if (threadIdx.x < dist && s_maxResult[threadIdx.x] < s_maxResult[compIdx])
		{
			s_maxResult[threadIdx.x] = s_maxResult[compIdx];
			s_maxIndexX[threadIdx.x] = compIdx;
			s_maxIndexY[threadIdx.x] = s_maxIndexY[compIdx];
		}
		__syncthreads();
	}
	int posX = s_maxIndexX[0];
	int posY = s_maxIndexY[0];
	if (posX - WINDOW_SIZE < -1 || posY - WINDOW_SIZE < -1)		//beyond the bounds exception
		return;

	//init the window matrix
	d_results += (posY*RESULT_WIDTH + posX);	//move to window starting index

	__shared__ float winMat[WINDOW_SIZE*WINDOW_SIZE];

	if (threadIdx.x < WINDOW_SIZE)
	{
#pragma unroll
		for (int i = 0; i < WINDOW_SIZE; i++)
		{
			winMat[i*WINDOW_SIZE + threadIdx.x] = d_results[i*RESULT_WIDTH + threadIdx.x];
		}
	}
	__syncthreads();

	//calculate sum of rows, cols
	__shared__ float rowSum[WINDOW_SIZE];
	__shared__ float colSum[WINDOW_SIZE];

	float* pTarget = nullptr;
	int strideX = WINDOW_SIZE;
	int strideY = 1;
	int syncIdx = threadIdx.x % WINDOW_SIZE;
	if (threadIdx.x < WINDOW_SIZE)
	{
		pTarget = colSum;
		strideX = 1;
		strideY = WINDOW_SIZE;
	}
	else if (threadIdx.x >= blockDim.x - WINDOW_SIZE)
	{
		pTarget = rowSum;
	}
	if (pTarget != nullptr)
	{
		pTarget[syncIdx] = 0;
#pragma unroll
		for (int i = 0; i < WINDOW_SIZE; i++)
		{
			pTarget[syncIdx] += winMat[syncIdx*strideX + i*strideY];
		}
	}
	__syncthreads();

	//calculate relative position from center of integer coordinates
	pTarget = nullptr;
	syncIdx = threadIdx.x % (WINDOW_SIZE / 2);
	if (threadIdx.x < WINDOW_SIZE / 2)
	{
		pTarget = colSum;
	}
	else if (threadIdx.x >= (blockDim.x - WINDOW_SIZE / 2))
	{
		pTarget = rowSum;
	}
	if (pTarget != nullptr)
	{
		pTarget[syncIdx] = pTarget[(int)WINDOW_SIZE / 2 + 1 + syncIdx] - pTarget[syncIdx];
	}
	__syncthreads();

	if (threadIdx.x == 0)
	{
#pragma unroll
		for (int i = 1; i < WINDOW_SIZE / 2; i++)
		{
			colSum[0] += colSum[i];
			rowSum[0] += rowSum[i];
		}
	}
	__syncthreads();

	float rx = (((colSum[0] / colSum[WINDOW_SIZE / 2]) + 1) / 2) - 0.5f;
	float ry = (((rowSum[0] / rowSum[WINDOW_SIZE / 2]) + 1) / 2) - 0.5f;


	//calculate stdev
	__shared__ float colSqSum[WINDOW_SIZE];
	pTarget = nullptr;
	syncIdx = threadIdx.x % WINDOW_SIZE;
	int isSquare = 0;
	if (threadIdx.x < WINDOW_SIZE)
	{
		pTarget = colSum;
	}
	else if (threadIdx.x >= blockDim.x - WINDOW_SIZE);
	{
		pTarget = colSqSum;
		isSquare++;
	}
	if (pTarget != nullptr)
	{
#pragma unroll
		for (int i = 0; i < WINDOW_SIZE; i++)
		{
			float x = winMat[i*WINDOW_SIZE + syncIdx];
			pTarget[syncIdx] += isSquare ? x*x : x;
		}
	}
	__syncthreads();

	pTarget = nullptr;
	syncIdx = threadIdx.x % (WINDOW_SIZE / 2);
	if (threadIdx.x < WINDOW_SIZE / 2)
	{
		pTarget = colSum;
	}
	else if (threadIdx.x >= (blockDim.x - WINDOW_SIZE / 2))
	{
		pTarget = colSqSum;
	}
	if (pTarget != nullptr)
	{
		pTarget[syncIdx] += pTarget[(int)WINDOW_SIZE / 2 + 1 + syncIdx];
	}
	__syncthreads();

	if (threadIdx.x == 0)
	{
#pragma unroll
		for (int i = 1; i < WINDOW_SIZE / 2; i++)
		{
			colSum[0] += colSum[i];
			colSqSum[0] += colSqSum[i];
		}
		colSum[0] /= WINDOW_SIZE*WINDOW_SIZE;
		colSqSum[0] /= WINDOW_SIZE*WINDOW_SIZE;
	}
	__syncthreads();

	float stdev = sqrtf(fabsf(colSqSum[0] - colSum[0] * colSum[0]));

	if (threadIdx.x == 0)
		*d_matchPoints = make_float2((float)d_featurePositions->x + (float)posX + (rx / (stdev * 2)),
		(float)d_featurePositions->y + (float)posY + (ry / (stdev * 2)));

	return;
}

extern "C" void ExtractFeaturePositions(float* d_results, int featureSize, int2* d_featurePositions, int* d_ghostFeature, float2* d_matchPoints)
{
	ExtractFeaturePositionsKernel <<<featureSize, RESULT_WIDTH >>>(d_results, featureSize, d_featurePositions, d_ghostFeature, d_matchPoints);
}
