/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

	// Declare an array for share memory...
	__shared__ float partialSum[2 * BLOCK_SIZE];
	
	// Initialize some variables to access data...
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;

	// Validation to avoid load data outside of the input array...
	if(start + t < size)
		partialSum[t] = in[start + t];
	else
		partialSum[t] = 0.0f;
	
	// Same validation for the other position...
	if(start + blockDim.x + t < size)
		partialSum[blockDim.x + t] = in[start + blockDim.x + t];
	else
		partialSum[blockDim.x + t] = 0.0f;

	// Iterate through share memory to compute the sum...
	for (int stride = blockDim.x; stride > 0; stride /= 2){
		__syncthreads();  // Synchronize the share memory load and each iteration...
		if (t < stride)
			partialSum[t] += partialSum[t + stride];
	} 
	// Do not forget to synchronize last iteration...
	__syncthreads();

	// Copy back the result...
	out[blockIdx.x] = partialSum[0];
}
