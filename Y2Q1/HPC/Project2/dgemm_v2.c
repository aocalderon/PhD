#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void dgemm0(double *a, double *b, double *c, int n);
void dgemm1(double *a, double *b, double *c, int n, int BLOCK_SIZE);
void dgemm2(double *a, double *b, double *c, int n);
int dgemm3(double *a, double *b, double *c, int n);
double verification(double *a, double *b, double *c0, double *c1, int n);

int main(int argc, char* argv[]){
	int BLOCK_SIZE = atoi(argv[1]);
	int n = atoi(argv[2]);
	uint64_t t0;
	float t;
	struct timespec begin, end;
	double *a, *b;
	double *c0, *c1, *c2;
	double diff;
	int i;

	srand(time(NULL));
	a = (double *) calloc(sizeof(double), n * n);
	for(i = 0; i < n * n; i++){
		a[i] = rand() / 1000000.0;
		a[i] = 1;
	}
	b = (double *) calloc(sizeof(double), n * n);
	for(i = 0; i < n * n; i++){
		b[i] = rand() / 1000000.0;
		b[i] = 1;
	}
	c0 = (double *) calloc(sizeof(double), n * n);
	c1 = (double *) calloc(sizeof(double), n * n);
	c2 = (double *) calloc(sizeof(double), n * n);

	clock_gettime(CLOCK_MONOTONIC, &begin);
	dgemm0(a, b, c0, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\t", t);
	
	clock_gettime(CLOCK_MONOTONIC, &begin);
	dgemm1(a, b, c1, n, BLOCK_SIZE);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\t", t);	
	
	clock_gettime(CLOCK_MONOTONIC, &begin);
	dgemm3(a, b, c2, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\t", t);	
	
	diff = verification(a, b, c0, c2, n);
	printf("%f\n", diff);
	
	int j;
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++)
			printf("%f ",c0[i*n+j]);
	printf("\n");
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++)
			printf("%f ",c2[i*n+j]);
	printf("\n");

	return 0;
}

void dgemm0(double *a, double *b, double *c, int n){
	int i,j,k;
	for(k = 0; k < n; k++){
		for(i = 0; i < n; i++){
			register double r = a[i*n+k];
			for(j = 0; j < n; j++){
				c[i*n+j] += r * b[k*n+j];
			}
		}
	}
}

void dgemm1(double *a, double *b, double *c, int n, int BLOCK_SIZE){
	int i,j,k;
	int ii,jj,kk;
	for(k = 0; k < n; k+=BLOCK_SIZE){
		for(i = 0; i < n; i+=BLOCK_SIZE){
			for(j = 0; j < n; j+=BLOCK_SIZE){
				for(kk = k; kk < k+BLOCK_SIZE; kk++){
					for(ii = i; ii < i+BLOCK_SIZE; ii++){
						register double r = a[ii*n+kk];
						for(jj = j; jj < j+BLOCK_SIZE; jj++){
							c[ii*n+jj] += r * b[kk*n+jj];
						}
					}
				}
			}
		}
	}
}

void dgemm2(double *a, double *b, double *c, int n, int BLOCK_SIZE){
	int i,j,k;
	int ii,jj,kk;
	for(i = 0; i < n; i+=BLOCK_SIZE){
		for(j = 0; j < n; j+=BLOCK_SIZE){
			for(k = 0; k < n; k+=BLOCK_SIZE){
				for(ii = i; ii < i+BLOCK_SIZE; ii += 3){
					for (jj = j; jj < i+BLOCK_SIZE; jj += 3){
						register double c11 = c[ii*n+jj];
						register double c12 = c[ii*n+jj+1];
						register double c13 = c[ii*n+jj+2];
						register double c21 = c[(ii+1)*n+jj];
						register double c22 = c[(ii+1)*n+jj+1];
						register double c23 = c[(ii+1)*n+jj+2];
						register double c31 = c[(ii+2)*n+jj];
						register double c32 = c[(ii+2)*n+jj+1];
						register double c33 = c[(ii+2)*n+jj+2];
						for (kk = k; kk < i+BLOCK_SIZE; kk++){
							register double a1 = a[ii*n+kk];
							register double a2 = a[(ii+1)*n+kk];
							register double a3 = a[(ii+2)*n+kk];
							register double b1 = b[kk*n+jj];
							register double b2 = b[kk*n+jj+1];
							register double b3 = b[kk*n+jj+2];
							c11 += a1*b1;
							c12 += a1*b2; 
							c13 += a1*b3;
							c21 += a2*b1;
							c22 += a2*b2;
							c23 += a2*b3;
							c31 += a3*b1;
							c32 += a3*b2;
							c33 += a3*b3;
						}
						c[ii*n+jj] = c11;
						c[ii*n+jj+1] = c12;
						c[ii*n+jj+2] = c13;
						c[(ii+1)*n+jj] = c21;
						c[(ii+1)*n+jj+1] = c22;
						c[(ii+1)*n+jj+2] = c23;
						c[(ii+2)*n+jj] = c31;
						c[(ii+2)*n+jj+1] = c32;
						c[(ii+2)*n+jj+2] = c33;
					}
				}
			}
		}
	}
}

double verification(double *a, double *b, double *c0, double *c1, int n){
	double diff, maxA, maxB;
	int i;

	diff = fabs(c1[0] - c0[0]);
	maxA = fabs(a[0]);
	maxB = fabs(b[0]);
	for(i = 0; i < n * n; i++){
		if(fabs(c1[i] - c0[i]) > diff)
			diff = fabs(c1[i] - c0[i]);
		if(fabs(a[i]) > maxA)
			maxA = fabs(a[i]);
		if(fabs(b[i]) > maxB)
			maxB = fabs(b[i]);
	}
	return diff / (maxA * maxB);
}
