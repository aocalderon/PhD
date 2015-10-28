/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    // Getting the position of the current element...
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Initializing the accumulator...
    float s = 0;
    
    // Discard positions outside of the dimension of the final matrix...
    if(y > m - 1 || x > n - 1) return;
    
    // Multiplying row and column elements and adding to the accumulator...
    for(int i = 0; i < k; ++i){
		s += A[y * k + i] * B[i * n + x];
	}
	// Copying the result to the final matrix...
	C[y * n + x] = s; 
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = 16; // Use 16x16 thread blocks

    // INSERT CODE HERE
    // Setting the dimension of the grid ensuring that the dimension of the
    // final matrix will be covered...
    const dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
	const dim3 dim_grid(n/BLOCK_SIZE + 1, m/BLOCK_SIZE + 1, 1);

    // Invoke CUDA kernel -----------------------------------------------------

    // INSERT CODE HERE
    // Calling the kernel with the above-mentioned setting... 
	mysgemm<<<dim_grid, dim_block>>>(m, n, k, A, B, C);
}


