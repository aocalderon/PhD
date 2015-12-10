#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "bfe.h"
#include "kernel.cu"

int main(int argc,char *argv[]){
	if(argc != 4){
		printf("Usage: %s TIMESTAMP EPSILON MU", argv[0]);
		return 1;
	}
	const int TIMESTAMP = atoi(argv[1]);
	const int EPSILON = atoi(argv[2]);
	const int E2 = EPSILON * EPSILON;
	const int MU = atoi(argv[3]);
	cudaError_t cuda_ret;
	
	FILE *in;
	FILE *out;
	in = fopen("oldenburg.csv", "r");
	out = fopen("output.csv", "w");
	fprintf(out, "oid;time;lat;lon;grid_id\n");
	char line[1024];
	int n = 0;
	short time;
	int lat; int lon;
	int max_lat = INT_MIN; int min_lat = INT_MAX;
	int max_lon = INT_MIN; int min_lon = INT_MAX;
	int M = 0; 
	int N = 0; 
	while (fgets(line, 1024, in)){
		atoi(strtok(line, ";"));
		if(atoi(strtok(NULL, ";\n")) != TIMESTAMP) continue;
		lat = atoi(strtok(NULL, ";\n"));
		if(lat > max_lat) max_lat = lat;
		if(lat < min_lat) min_lat = lat;
		lon = atoi(strtok(NULL, ";\n"));
		if(lon > max_lon) max_lon = lon;
		if(lon < min_lon) min_lon = lon;
		n++;
	}
	int *x;
	x = (int*) malloc( sizeof(int) * n);
	int *y;
	y = (int*) malloc( sizeof(int) * n);
	int *g;
	g = (int*) malloc( sizeof(int) * n);
	int *i;
	i = (int*) malloc( sizeof(int) * n);
	printf("Min and max latitude:\t(%d, %d)\n", min_lat, max_lat);
	printf("Min and max longitude:\t(%d, %d)\n", min_lon, max_lon);
	M = (max_lat - min_lat) / EPSILON + 1;
	N = (max_lon - min_lon) / EPSILON + 1;
	rewind(in);
	int j = 0;
	while (fgets(line, 1024, in)){
		atoi(strtok(line, ";"));
		time = atoi(strtok(NULL, ";\n"));
		if(time != TIMESTAMP) continue;
		lat = atoi(strtok(NULL, ";\n"));
		lon = atoi(strtok(NULL, ";\n"));
		g[j] = M * ((N - 1) - ((lon - min_lon) / EPSILON)) + ((lat - min_lat) / EPSILON);
		x[j] = lat;
		y[j] = lon;
		i[j] = j; 
       	//printf("%d;%hi;%d;%d;%d\n", oid, time, lat, lon, g[j]);
		j++;
	}
	printf("Number of points:\t%d\n", n);
	printf("M x N : %d x %d\n", M, N);
	int c = M * N;
	//int r = createGrid("grid.shp", EPSILON, min_lat, max_lat, min_lon, max_lon);
	printf("Sorting arrays...\n");
	thrust::device_vector<int> d_x(x, x + n);
	thrust::device_vector<int> d_y(y, y + n);
	thrust::device_vector<int> d_g(g, g + n);
	thrust::device_vector<int> d_i(i, i + n);
	thrust::sort_by_key(d_g.begin(), d_g.end(), d_i.begin());
	thrust::gather(d_i.begin(), d_i.end(), d_x.begin(), d_x.begin());
	thrust::gather(d_i.begin(), d_i.end(), d_y.begin(), d_y.begin());
	thrust::copy(d_g.begin(), d_g.end(), g);
	thrust::copy(d_i.begin(), d_i.end(), i);
	thrust::copy(d_x.begin(), d_x.end(), x);
	thrust::copy(d_y.begin(), d_y.end(), y);

	printf("Counting point indices...\n");
	int *a;
	a = (int*) malloc(sizeof(int) * c);
	int *b;
	b = (int*) malloc(sizeof(int) * c);
	a[0] = g[0];
	b[0] = 0;
	int k = 0;
	for(j = 0; j < n; j++){
		if(g[j] != a[k]){
			k++;
			a[k] = g[j];
			b[k] = j;
		}	
	}
	b[++k] = n;

	int *x_d, *y_d, *g_d;
	int *a_d, *b_d;
	unsigned long *N_DISKS;
	unsigned long *result;

	result = (unsigned long*) malloc(sizeof(long) * c);

	printf("cudaMalloc and cudaMemcpy stage...\n");
	cuda_ret = cudaMalloc((void **) &x_d, sizeof(int) * n);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMalloc for x ...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cuda_ret = cudaMalloc((void **) &y_d, sizeof(int) * n);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMalloc for y ...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cuda_ret = cudaMalloc((void **) &g_d, sizeof(int) * n);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMalloc for g ...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cuda_ret = cudaMalloc((void **) &N_DISKS, sizeof(long) * c);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMalloc for N_DISKS ...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cuda_ret = cudaMalloc((void **) &a_d, sizeof(int) * c);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMalloc for a ...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cuda_ret = cudaMalloc((void **) &b_d, sizeof(int) * c);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMalloc for b ...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();
	cuda_ret = cudaMemcpy(x_d, x, sizeof(int) * n, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMemcpy for x_d...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cuda_ret = cudaMemcpy(y_d, y, sizeof(int) * n, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMemcpy for y_d...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cuda_ret = cudaMemcpy(g_d, g, sizeof(int) * n, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMemcpy for g_d...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cuda_ret = cudaMemcpy(a_d, a, sizeof(int) * c, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMemcpy for a_d...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cuda_ret = cudaMemcpy(b_d, b, sizeof(int) * c, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMemcpy for b_d...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();
	//const dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
	//const dim3 dim_grid(((n - 1) / BLOCK_SIZE) + 1, ((m - 1) / BLOCK_SIZE) + 1, 1);
	const dim3 grid(1, 1, 1);
	const dim3 block(k, 1, 1);

	// Calling the kernel... 
	printf("Running the kernel...\nk=%d\n", k);
	parallelBFE<<<grid, block>>>(x_d, y_d, g_d, a_d, b_d, n, k, M, N, E2, N_DISKS);
	
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess){
		printf("\nError lunching kernel...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	cuda_ret = cudaMemcpy(result, N_DISKS, sizeof(long) * c, cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMemcpy for result...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();
	
	printf("\n");
	for(int j = 0; j < k; j++){
		if(j % M == 0) printf("\n");	
		if(result[j] >= MU){
			printf("%3d->%3li  ", a[j], result[j]);		
		}
	}
	printf("\n");
	/*
	int BLOCK_SIZE = 4;
	const dim3 grid2(BLOCK_SIZE, BLOCK_SIZE, 1);
	const dim3 block2(BLOCK_SIZE / N, BLOCK_SIZE / M, 1);

	// Calling other the kernel... 
	printf("Running the kernel...\nk=%d\n", k);
	seeThreadIndex<<<grid2, block2>>>(a_d, b_d, k, M, N, N_DISKS);
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess){
		printf("\nError lunching kernel...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	cuda_ret = cudaMemcpy(result, N_DISKS, sizeof(long) * c, cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess){
		printf("\nChecking cudaMemcpy for result...  %s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();
	printf("\n");
	for(int j = 0; j < k; j++){
		if(j % M == 0) printf("\n");	
		if(result[j] >= MU){
			printf("%3li  ", result[j]);		
		}
	}
	printf("\n");
	*/

	cudaFree(x_d);

	cudaFree(y_d);
	cudaFree(g_d);
	cudaFree(N_DISKS);
	
	free(x);
	free(y);
	free(g);
	free(i);
	free(result);
	
	return 0;
}
