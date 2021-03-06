#pragma once

#include "cuda_runtime.h"
#include "stdafx.h"

#define blockSize 16
#define threadSize 16 // manipulate heuristically

__global__ void MatMul(const Mat A, const Mat B, Mat res)
{
	int row = (blockIdx.y*blockDim.y + threadIdx.y) * res.col ;
	int col = blockIdx.x*blockDim.x + threadIdx.x ;

	if(row >= A.row || col >= B.col)
		return;

	__shared__ float localSum[blockSize][blockSize]; // this variable could be async
	localSum[threadIdx.y][threadIdx.x] = 0; // initialize
	int iterNum = A.col/threadSize;
	iterNum = (A.col%threadSize == 0)? iterNum : iterNum+1 ;
	for(int i=0; i<iterNum; i++)
	{
		__shared__ float Asub[blockSize][threadSize];
		__shared__ float Bsub[threadSize][blockSize];
		Asub[threadIdx.y][threadIdx.x] = *((float*)A.elements + row*A.col + i*threadSize + threadIdx.x);
		Bsub[threadIdx.y][threadIdx.x] = *((float*)B.elements + (i*threadSize + threadIdx.y)*B.col + col);
		
		syncthreads();
		
		int subSize = threadSize;
		if(i == iterNum-1)  // last loop exception
			subSize = A.col - i*threadSize;

		for(int j=0; j<subSize; j++)
		{
			localSum[threadIdx.y][threadIdx.x] += Asub[threadIdx.y][j] * Bsub[j][threadIdx.x];
		}
	}
	
	*((float*)res.elements + row*res.col + col) = localSum[threadIdx.y][threadIdx.x];
}

extern "C" void MatMulGPU(const Mat A, const Mat B, Mat res)
{
	
	Mat d_A, d_B, d_res;
	int sizeA = A.row*A.col*sizeof(float);
	int sizeB = B.row*B.col*sizeof(float);
	int sizeRes = A.row*B.col*sizeof(float);
	d_A.row = A.row; d_A.col = A.col;
	cudaMalloc(&d_A.elements, sizeA);
	cudaMemcpy(d_A.elements, A.elements, sizeA, cudaMemcpyHostToDevice);
	d_B.row = B.row; d_B.col = B.col;
	cudaMalloc(&d_B.elements, sizeB);
	cudaMemcpy(d_B.elements, B.elements, sizeB, cudaMemcpyHostToDevice);
	d_res.row = A.row; d_res.col = B.col;
	cudaMalloc(&d_res.elements, sizeRes);
	
	//float tmpSize = (float)(A.col * B.row * res.col * res.row) / (float)blockSize;
	//int blocksPerGrid = (tmpSize - (int)tmpSize) > 0 ? tmpSize+1 : tmpSize;
	dim3 blocksPerGrid(res.col/blockSize+1, res.row/blockSize+1);
	dim3 threadsPerBlock(blockSize, blockSize);

	MatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_res);
	cudaMemcpy(res.elements, d_res.elements, sizeRes, cudaMemcpyDeviceToHost);
	
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_res.elements);
}