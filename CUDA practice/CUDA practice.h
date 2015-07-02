#pragma once

#ifndef __CUDA_PRAC_H
#define __CUDA_PRAC_H
#endif

#include "cuda_runtime.h"
#include "stdafx.h"

extern "C" void MatMulGPU(const Mat A, const Mat B, Mat res);
extern "C" void MatMulGPU_Texture(const Mat A, const Mat B, Mat res);
extern "C" void ExtractFeaturePositions(float* d_results, int featureSize, int2* d_featurePositions, int* d_ghostFeature, float2* d_matchPoints);

void MatTranspose(const Mat src, Mat dst)
{
	int iterNum = src.row*src.col;

	for (int n = 0; n < iterNum; n++)
	{
		int i = n / src.col;
		int j = n%src.col;
		*((float*)dst.elements + n) = *((float*)src.elements + j*src.col + i);
	}
}

void MatMulCPU(const Mat A, const Mat B, Mat res)
{
	for (int i = 0; i < A.row; i++)
	{
		for (int j = 0; j < B.col; j++)
		{
			float sum = 0;
			for (int k = 0; k < A.col; k++)
			{
				sum += *((float*)A.elements + i*A.col + k) * *((float*)B.elements + k*B.col + j);
			}
			*((float*)res.elements + i*B.col + j) = sum;
		}
	}
}