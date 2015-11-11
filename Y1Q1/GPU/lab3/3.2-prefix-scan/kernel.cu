/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

__global__ void scan(float *out, float *in, unsigned size){
	__shared__ float section[2 * BLOCK_SIZE];
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if(t < size)
		if(t == 0)
			section[0] = 0.0f;
		else
			section[threadIdx.x] = in[t - 1];
	__syncthreads();
	
	for(int stride = 1; stride <= BLOCK_SIZE; stride = stride * 2){
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if(index < 2 * BLOCK_SIZE)
			section[index] += section[index - stride];
		__syncthreads();
	}
	
	for(int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2){
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if(index + stride < 2 * BLOCK_SIZE)
			section[index + stride] += section[index];
		__syncthreads();
	}
	//__syncthreads();
	if(t < size) 
		out[t] = section[threadIdx.x];
}

__global__ void post(float *out, float *n, unsigned size){
	int t = blockDim.x * blockIdx.x + threadIdx.x;
	
	out[t] += n[t / BLOCK_SIZE];
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned size){
	dim3 dim_block(BLOCK_SIZE, 1, 1);
	dim3 dim_grid(size/BLOCK_SIZE + 1, 1, 1);
	scan<<<dim_grid, dim_block>>>(out, in, size);
}

void postScan(float *out, float *n, unsigned size){
	dim3 dim_block(BLOCK_SIZE, 1, 1);
	dim3 dim_grid(size/BLOCK_SIZE + 1, 1, 1);
	post<<<dim_grid, dim_block>>>(out, n, size);
}
