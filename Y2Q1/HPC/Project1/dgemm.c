#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Declaring functions...
void dgemm0(double *a, double *b, double *c, int n);
void dgemm1(double *a, double *b, double *c, int n);
void dgemm2(double *a, double *b, double *c, int n);
void dgemm3(double *a, double *b, double *c, int n);
void dgemm4(double *a, double *b, double *c, int n);
double verification(double *a, double *b, double *c0, double *c1, int n);

int main(int argc, char* argv[]){
	// Reading N from command line...
	int n = atoi(argv[1]);
	uint64_t t0;
	float t;
	struct timespec begin, end;
	double *a, *b;
	double *c0, *c1, *c2, *c3, *c4;
	double diff;
	int i;

	// Random seed...
	srand(time(NULL));
	// Creating matrices A and B...
	a = (double *) calloc(sizeof(double), n * n);
	for(i = 0; i < n * n; i++){
		a[i] = rand() / 1000000.0;
	}
	b = (double *) calloc(sizeof(double), n * n);
	for(i = 0; i < n * n; i++){
		b[i] = rand() / 1000000.0;
	}
	// Allocating memory for matrices C's...
	c0 = (double *) calloc(sizeof(double), n * n);
	c1 = (double *) calloc(sizeof(double), n * n);
	c2 = (double *) calloc(sizeof(double), n * n);
	c3 = (double *) calloc(sizeof(double), n * n);
	c4 = (double *) calloc(sizeof(double), n * n);

	// Running dgemm0...
	clock_gettime(CLOCK_MONOTONIC, &begin);
	dgemm0(a, b, c0, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\t", t);

	// Running dgemm1...
	clock_gettime(CLOCK_MONOTONIC, &begin);
	dgemm1(a, b, c1, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\t", t);
        
	// Running dgemm2...
	clock_gettime(CLOCK_MONOTONIC, &begin);
	dgemm2(a, b, c2, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\t", t);

	// Running dgemm3...
	clock_gettime(CLOCK_MONOTONIC, &begin);
	dgemm3(a, b, c3, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\t", t);

	// Running dgemm4...
	clock_gettime(CLOCK_MONOTONIC, &begin);
	dgemm3(a, b, c4, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
	t = t0 / 1000000000.0;
	printf("%f\t", t);

	// Running verifications...
	diff = verification(a, b, c0, c1, n);
	printf("%f\t", diff);
	diff = verification(a, b, c0, c2, n);
	printf("%f\t", diff);
	diff = verification(a, b, c0, c3, n);
	printf("%f\t", diff);
	diff = verification(a, b, c0, c4, n);
	printf("%f\n", diff);

	return 0;
}

void dgemm0(double *a, double *b, double *c, int n){
	int i,j,k;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			for(k = 0; k < n; k++){
				c[i*n+j] += a[i*n+k] * b[k*n+j];
			}
		}
	}
}

void dgemm1(double *a, double *b, double *c, int n){
	int i,j,k;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			register double r = c[i*n+j];
			for(k = 0; k < n; k++){
				r += a[i*n+k] * b[k*n+j];
			}
			c[i*n+j] = r;
		}
	}
}

void dgemm2(double *a, double *b, double *c, int n) {
	int i, j, k;
	for (i = 0; i < n; i+=2){
		for (j = 0; j < n; j+=2){
			for (k = 0; k < n; k+=2){
				c[i*n+j] = a[i*n+k] * b[k*n+j] + a[i*n+(k+1)] * b[(k+1)*n+j] + c[i*n+j];
				c[(i+1)*n+j] = a[(i+1)*n+k] * b[k*n+j] + a[(i+1)*n+(k+1)] * b[(k+1)*n+j] + c[(i+1)*n+j];
				c[i*n+(j+1)] = a[i*n+k] * b[k*n+(j+1)] + a[i*n+(k+1)] * b[(k+1)*n+(j+1)] + c[i*n+(j+1)];
				c[(i+1)*n+(j+1)] = a[(i+1)*n+k] * b[k*n+(j+1)] + a[(i+1)*n+(k+1)] * b[(k+1)*n+(j+1)] + c[(i+1)*n+(j+1)];
			}
		}
	}
}

void dgemm3(double *a, double *b, double *c, int n) {
	int i, j, k;
	for (i = 0; i < n; i+=2){
		for (j = 0; j < n; j+=2){
			register double cc0 = c[i*n+j];
			register double cc1 = c[(i+1)*n+j];
			register double cc2 = c[i*n+(j+1)];
			register double cc3 = c[(i+1)*n+(j+1)];
			for (k = 0; k < n; k+=2){
				register double aa0 = a[i*n+k];
				register double aa1 = a[i*n+(k+1)];
				register double aa2 = a[(i+1)*n+k];
				register double aa3 = a[(i+1)*n+(k+1)];
				register double bb0 = b[k*n+j];
				register double bb1 = b[(k+1)*n+j];
				register double bb2 = b[k*n+(j+1)];
				register double bb3 = b[(k+1)*n+(j+1)];
				cc0 += aa0 * bb0 + aa1 * bb1;
				cc1 += aa2 * bb0 + aa3 * bb1;
				cc2 += aa0 * bb2 + aa1 * bb3;
				cc3 += aa2 * bb2 + aa3 * bb3;
			}
			c[i*n+j] = cc0;
			c[(i+1)*n+j] = cc1;
			c[i*n+(j+1)] = cc2;
			c[(i+1)*n+(j+1)] = cc3;
		}
	}
}

void dgemm4(double *a, double *b, double *c, int n) {
	int i, j, k;
	for(i = 0; i < n; i += 3){
		for(j = 0; j < n; j += 3){
			register double c0 = c[i*n+j];
			register double c1 = c[i*n+j+1];
			register double c2 = c[i*n+j+2];
			register double c3 = c[(i+1)*n+j];
			register double c4 = c[(i+1)*n+(j+1)];
			register double c5 = c[(i+1)*n+(j+2)];
			register double c6 = c[(i+2)*n+j];
			register double c7 = c[(i+2)*n+(j+1)];
			register double c8 = c[(i+2)*n+(j+2)];
			for(k = 0; k < n; k++){
				register double a0 = a[i*n+k];
				register double a1 = a[(i+1)*n+k];
				register double a2 = a[(i+2)*n+k];
				register double b0 = b[k*n+j];
				register double b1 = b[k*n+(j+1)];
				register double b2 = b[k*n+(j+2)];
				c0 += a0*b0;
				c1 += a0*b1; 
				c2 += a0*b2;
				c3 += a1*b0;
				c4 += a1*b1;
				c5 += a1*b2;
				c6 += a2*b0;
				c7 += a2*b1;
				c8 += a2*b2;
			}
			c[i*n+j] 		= c0;
			c[i*n+j+1] 		= c1;
			c[i*n+j+2] 		= c2;
			c[(i+1)*n+j] 	= c3;
			c[(i+1)*n+(j+1)]= c4;
			c[(i+1)*n+(j+2)]= c5;
			c[(i+2)*n+j] 	= c6;
			c[(i+2)*n+(j+1)]= c7;
			c[(i+2)*n+(j+2)]= c8;
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
