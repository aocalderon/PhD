
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char* argv[]){
	int n = atoi(argv[1]);
	uint64_t t0;
	float t;
	struct timespec begin, end;
	double *a, *b;
	double *c;
	int *a1, *b1, *c1; 
	double diff;
	int CACHE_SIZE = 4;
	int LINE_SIZE = 4;
	int i,j,k;

	srand(time(NULL));
	a = (double *) calloc(sizeof(double), n * n);
	for(i = 0; i < n * n; i++){
		a[i] = rand() / 1000000.0;
	}
	b = (double *) calloc(sizeof(double), n * n);
	for(i = 0; i < n * n; i++){
		b[i] = rand() / 1000000.0;
	}
	c = (double *) calloc(sizeof(double), n * n);
	for(i = 0; i < n * n; i++){
		c[i] = 0.0;
	}
	a1 = (int *) calloc(sizeof(int), n * n);
	for(i = 0; i < n * n; i++){
		a1[i] = 0;
	}
	b1 = (int *) calloc(sizeof(int), n * n);
	for(i = 0; i < n * n; i++){
		b1[i] = 0;
	}
	c1 = (int *) calloc(sizeof(int), n * n);
	for(i = 0; i < n * n; i++){
		c1[i] = 0;
	}

	clock_gettime(CLOCK_MONOTONIC, &begin);
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			register double r = c[i*n+j];
			c1[i*n+j]++; 
			printf("Reading C[%d,%d] %d\n",(i*n+j)/n, j, c1[i*n+j]);
			for(k = 0; k < n; k++){
				r += a[i*n+k] * b[k*n+j];
				a1[i*n+k]++;
				b1[k*n+j]++;
				printf("Reading A[%d,%d] %d\n",(i*n+k)/n, k, a1[i*n+k]);
				printf("Reading B[%d,%d] %d\n",(k*n+j)/n, j, b1[k*n+j]);
			}
			c[i*n+j] = r;
		}
	}
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			printf("Pos (%d,%d) -> C: %d \t A: %d \t B: %d\n",i,j,c1[i*n+j],a1[i*n+j],b1[i*n+j]);
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\n", t);
	
	/**********************************************************/
	for(i = 0; i < n * n; i++){
		a1[i] = 0;
	}
	b1 = (int *) calloc(sizeof(int), n * n);
	for(i = 0; i < n * n; i++){
		b1[i] = 0;
	}
	c1 = (int *) calloc(sizeof(int), n * n);
	for(i = 0; i < n * n; i++){
		c1[i] = 0;
	}
	
	clock_gettime(CLOCK_MONOTONIC, &begin);
	for(k = 0; k < n; k++){
		for(i = 0; i < n; i++){
			register double r = a[i*n+k];
			a1[i*n+k]++;
			printf("Reading A[%d,%d] %d\n",(i*n+k)/n, k, a1[i*n+k]);
			for(j = 0; j < n; j++){
				c[i*n+j] += r * b[k*n+j];
				c1[i*n+j]++; 
				b1[k*n+j]++;
				printf("Reading C[%d,%d] %d\n",(i*n+j)/n, j, c1[i*n+j]);
				printf("Reading B[%d,%d] %d\n",(k*n+j)/n, j, b1[k*n+j]);
			}
		}
	}
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			printf("Pos (%d,%d) -> C: %d \t A: %d \t B: %d\n",i,j,c1[i*n+j],a1[i*n+j],b1[i*n+j]);
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\n", t);

	return 0;
}

