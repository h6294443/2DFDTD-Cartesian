#include "graphics.h"

texture<float4, 2, cudaReadModeElementType> inTex;

__constant__ unsigned int  dvrgb[256];

__global__ void find_min_and_max_on_gpu(int nblocks, float* field, 
										float* minimum_field_value, 
										float* maximum_field_value)
{
	__shared__ float minarr[1024];
	__shared__ float maxarr[1024];

	int i = threadIdx.x;
	int nTotalThreads = blockDim.x;

	minarr[i] = field[i];
	maxarr[i] = minarr[i];
	for (int j = 1; j<nblocks; j++)
	{
		minarr[i + nTotalThreads] = field[i + nTotalThreads*j];
		if (minarr[i] > minarr[i + nTotalThreads])
			minarr[i] = minarr[i + nTotalThreads];

		if (maxarr[i] < minarr[i + nTotalThreads])
			maxarr[i] = minarr[i + nTotalThreads];
		__syncthreads();
	}
	__syncthreads();

	while (nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		if (threadIdx.x < halfPoint)
		{
			float temp = minarr[i + halfPoint];

			if (temp < minarr[i]) minarr[i] = temp;

			temp = maxarr[i + halfPoint];
			if (temp > maxarr[i]) maxarr[i] = temp;
		}
		__syncthreads();
		nTotalThreads = (nTotalThreads >> 1);
	}
	if (i == 0)
	{
		minimum_field_value[0] = minarr[0];
		maximum_field_value[0] = maxarr[0];
	}
}

void createColormapOnGpu()
{
	cudaError_t et;
	et = cudaMemcpyToSymbol(dvrgb, rgb, 256 * sizeof(int), 0, cudaMemcpyHostToDevice);
}

__global__ void
create_image_on_gpu(unsigned int* g_odata, float* Ez, int M, float minval, float maxval)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int cind;
	float F;

	int ci = j*M + i;
	int ti = (j + 1)*M + i;
	if (j == M - 1) ti = (j)*M + i;
	F = Ez[ti] - minval;
	cind = floor(255 * F / (maxval - minval));
	if (cind > 255) cind = 255;
	g_odata[ci] = dvrgb[cind];
}

void createImageOnGpu(unsigned int* g_odata)
{
	dim3 block(TILE_SIZE, TILE_SIZE, 1);
	dim3 grid(M / block.x, N / block.y, 1);
	dim3 gridm = dim3(1, 1, 1);
	dim3 blockm = dim3(TILE_SIZE*TILE_SIZE, 1, 1);
	int  nblocks = grid.x * grid.y;
	float minval;
	float maxval;
	float *dvF;

	//if (show_Ez) dvF = dev_ez_float; else dvF = dev_hx_float;
	dvF = dev_ez_float;

	find_min_and_max_on_gpu << < gridm, blockm >> >(nblocks, dvF, dvminimum_field_value, dvmaximum_field_value);

	cudaMemcpy(&minval, dvminimum_field_value, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&maxval, dvmaximum_field_value, sizeof(float), cudaMemcpyDeviceToHost);

	if (minval>0.0) minval = 0.0;
	if (maxval<0.0) maxval = 0.0;
	if (abs(minval)>maxval) maxval = -minval; else minval = -maxval;
	if (minval<global_min_field) global_min_field = minval;
	if (maxval>global_max_field) global_max_field = maxval;

	//minval = -1.0;	maxval = 1.0;	global_min_field = -1.0; global_max_field = 1.0;
	
	create_image_on_gpu << < grid, block >> >(g_odata, dvF, M, global_min_field, global_max_field);
}