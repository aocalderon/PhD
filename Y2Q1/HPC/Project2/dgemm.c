#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void dgemm0(double *a, double *b, double *c, int n);
void dgemm1(double *a, double *b, double *c, int n, int BLOCK_SIZE);
void dgemm2(double *a, double *b, double *c, int n, int BLOCK_SIZE);
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
	dgemm2(a, b, c2, n, BLOCK_SIZE);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\t", t);	
	
	diff = verification(a, b, c0, c1, n);
	printf("%f\t", diff);
	diff = verification(a, b, c0, c2, n);
	printf("%f\n", diff);
	
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
				for (ii = i; ii < i+BLOCK_SIZE; ii+=2){
					for (jj = j; jj < j+BLOCK_SIZE; jj+=2){
						register double cc0 = c[ii*n+jj];
						register double cc1 = c[(ii+1)*n+jj];
						register double cc2 = c[ii*n+(jj+1)];
						register double cc3 = c[(ii+1)*n+(jj+1)];
						for (kk = k; kk < k+BLOCK_SIZE; kk+=2){
							register double aa0 = a[ii*n+kk];
							register double aa1 = a[ii*n+(kk+1)];
							register double aa2 = a[(ii+1)*n+kk];
							register double aa3 = a[(ii+1)*n+(kk+1)];
							register double bb0 = b[kk*n+jj];
							register double bb1 = b[(kk+1)*n+jj];
							register double bb2 = b[kk*n+(jj+1)];
							register double bb3 = b[(kk+1)*n+(jj+1)];
							cc0 += aa0 * bb0 + aa1 * bb1;
							cc1 += aa2 * bb0 + aa3 * bb1;
							cc2 += aa0 * bb2 + aa1 * bb3;
							cc3 += aa2 * bb2 + aa3 * bb3;
						}
						c[ii*n+jj] = cc0;
						c[(ii+1)*n+jj] = cc1;
						c[ii*n+(jj+1)] = cc2;
						c[(ii+1)*n+(jj+1)] = cc3;
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
