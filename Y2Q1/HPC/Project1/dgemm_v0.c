#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void dgemm0(double *a, double *b, double *c, int n);
void dgemm1(double *a, double *b, double *c, int n);
void dgemm2(double *a, double *b, double *c, int n);
double verification(double *a, double *b, double *c0, double *c1, int n);

int main(int argc, char* argv[]){
	int n = atoi(argv[1]);
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
	}
        b = (double *) calloc(sizeof(double), n * n);
        for(i = 0; i < n * n; i++){
                b[i] = rand() / 1000000.0;
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
        dgemm1(a, b, c1, n);
        clock_gettime(CLOCK_MONOTONIC, &end);
        t0 = 1000000000L * (end.tv_sec - begin.tv_sec) + end.tv_nsec - begin.tv_nsec;
        t = t0 / 1000000000.0;
		printf("%f\t", t);
        
	clock_gettime(CLOCK_MONOTONIC, &begin);
	dgemm2(a, b, c2, n);
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
				c[i*n+j] = a[i*n+k] * b[k*n+j] + a[i*n+k+1] * b[(k+1)*n+j] + c[i*n+j];
				c[(i+1)*n+j] = a[(i+1)*n+k] * b[k*n+j] + a[(i+1)*n+(k+1)] * b[(k+1)*n+j] + c[(i+1)*n+j];
				c[i*n+(j+1)] = a[i*n+k] * b[k*n+(j+1)] + a[i*n+(k+1)] * b[(k+1)*n+(j+1)] + c[i*n+(j+1)];
				c[(i+1)*n+(j+1)] = a[(i+1)*n+k] * b[k*n+(j+1)] + a[(i+1)*n+(k+1)] * b[(k+1)*n+(j+1)] + c[(i+1)*n+(j+1)];
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
