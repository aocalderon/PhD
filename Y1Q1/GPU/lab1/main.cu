/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

int main (int argc, char *argv[])
{
    Timer timer;
    cudaError_t cuda_ret;
    cudaError_t err;

    // Initialize host variables ----------------------------------------------
    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    } else if (argc == 2) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    } else if (argc == 4) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./sgemm                # All matrices are 1000 x 1000"
           "\n    Usage: ./sgemm <m>            # All matrices are m x m"
           "\n    Usage: ./sgemm <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
           "\n");
        exit(0);
    }

    A_sz = matArow*matAcol;
    B_sz = matBrow*matBcol;
    C_sz = matArow*matBcol;

    A_h = (float*) malloc( sizeof(float) * A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float) * B_sz );
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float) * C_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol, matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------
    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    // INSERT CODE HERE
    // cudaMalloc calls to allocate memory in the device...
	err = cudaMalloc((void**) &A_d, sizeof(float) * A_sz);
	if (err!=cudaSuccess) {
		printf("\nChecking cudaMalloc for A... %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	} 
	err = cudaMalloc((void**) &B_d, sizeof(float) * B_sz);
	if (err!=cudaSuccess) {
		printf("\nChecking cudaMalloc for B... %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	} 
	err = cudaMalloc((void**) &C_d, sizeof(float) * C_sz);
	if (err!=cudaSuccess) {
		printf("\nChecking cudaMalloc for C... %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	} 
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    // INSERT CODE HERE
    // cudaMemcpy calls to send variables to the device...
	err = cudaMemcpy(A_d, A_h, sizeof(float) * A_sz, cudaMemcpyHostToDevice);
	if (err!=cudaSuccess) {
		printf("Checking cudaMemcpy for A... %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	} 
	err = cudaMemcpy(B_d, B_h, sizeof(float) * B_sz, cudaMemcpyHostToDevice);
	if (err!=cudaSuccess) {
		printf("Checking cudaMemcpy for B... %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	} 
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard sgemm interface
    // Let's run the kernel...
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    basicSgemm('N', 'N', matArow, matBcol, matBrow, 1.0f, A_d, matArow, B_d, matBrow, 0.0f, C_d, matBrow);
    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables to host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    // INSERT CODE HERE
    // Sending back the results...
	err = cudaMemcpy(C_h, C_d, sizeof(float) * C_sz, cudaMemcpyDeviceToHost);
	if (err!=cudaSuccess) {
		printf("Got it!!! %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	} 
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------
    printf("Verifying results..."); fflush(stdout);
    // verify(A_h, B_h, C_h, matArow, matAcol, matBcol);
    
    // Printing relatively small matrices just for testing purposes...
    if(matArow * matBcol < 100){ 
		for(int i = 0; i < A_sz; ++i){
			if(i % matAcol == 0){
				printf("\n");
			} 
			printf("%.2f\t", A_h[i]);
		}
		printf("\n");
		for(int i = 0; i < B_sz; ++i){
			if(i % matBcol == 0){
				printf("\n");
			} 
			printf("%.2f\t", B_h[i]);
		}
		printf("\n");
		for(int i = 0; i < C_sz; ++i){
			if(i % matBcol == 0){
				printf("\n");
			} 
			printf("%.2f\t", C_h[i]);
		}
		printf("\n");
	}
	
    // Free memory ------------------------------------------------------------
    free(A_h);
    free(B_h);
    free(C_h);

    // INSERT CODE HERE
    // cudaFree calls to free memory...
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

    return 0;
}

