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

// INSERT KERNEL(S) HERE

__global__ void scan(float *out, float *in, unsigned size){
	__shared__ float section[2 * BLOCK_SIZE];
	
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if(t < size)
		section[threadIdx.x] = in[t];
	__syncthreads();
	
	section[threadIdx.x]++;
	out[threadIdx.x] = section[threadIdx.x];
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned size)
{
    // INSERT CODE HERE
	dim3 dim_block(BLOCK_SIZE, 1, 1);
	dim3 dim_grid(size/BLOCK_SIZE + 1, 1, 1);
	scan<<<dim_grid, dim_block>>>(out, in, size);
}

