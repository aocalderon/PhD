#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BILLION 1000000000L

int RandMatrixGen(double *M, int n)
{
	int i;
	srand (time(NULL));
	for (i = 0; i < n*n; i++)
	  M[i] =  2.0*rand()/RAND_MAX - 1.0;
	return 0;
}

int dgemm0(double *a, double *b, double *c, int n)
{
	int i, j, k;
	for (i = 0; i < n; i++)
	  for (j = 0; j < n; j++)
		for (k = 0; k < n; k++)
		  c[i*n+j] += a[i*n+k] * b[k*n+j];
	return 0;
}

int dgemm1(double *a, double *b, double *c, int n)
{
	int i, j, k;
	for (i = 0; i < n; i++)
	  for (j = 0; j < n; j++)
	  {
		  register double r = c[i*n+j];
		  for (k = 0; k < n; k++)
			r += a[i*n+k] * b[k*n+j];
		  c[i*n+j] = r;
	  }
	return 0;
}

int dgemm2(double *a, double *b, double *c, int n)
{
	int i, j, k;
	for (i = 0; i < n; i += 2)
	  for (j = 0; j < n; j += 2)
	  {
		  register double c1 = c[i*n+j];
		  register double c2 = c[i*n+j+1];
		  register double c3 = c[(i+1)*n+j];
		  register double c4 = c[(i+1)*n+j+1];
		for (k = 0; k < n; k += 2)
		{
			register double a1 = a[i*n+k];
			register double a2 = a[i*n+k+1];
			register double a3 = a[(i+1)*n+k];
			register double a4 = a[(i+1)*n+k+1];
			register double b1 = b[k*n+j];
			register double b2 = b[(k+1)*n+j];
			register double b3 = b[k*n+j+1];
			register double b4 = b[(k+1)*n+j+1];
			c1 += a1*b1 + a2*b2;
			c2 += a1*b3 + a2*b4;
			c3 += a3*b1 + a4*b2;
			c4 += a3*b3 + a4*b4;
		}
		c[i*n+j] = c1;
		c[i*n+j+1] = c2;
		c[(i+1)*n+j] = c3;
		c[(i+1)*n+j+1] = c4;
	  }
	return 0;
}

int dgemm3(double *a, double *b, double *c, int n)
{
	int i, j, k;
	for (i = 0; i < n; i += 3)
	  for (j = 0; j < n; j += 3)
	  {
		  register double c11 = c[i*n+j];
		  register double c12 = c[i*n+j+1];
		  register double c13 = c[i*n+j+2];
		  register double c21 = c[(i+1)*n+j];
		  register double c22 = c[(i+1)*n+j+1];
		  register double c23 = c[(i+1)*n+j+2];
		  register double c31 = c[(i+2)*n+j];
		  register double c32 = c[(i+2)*n+j+1];
		  register double c33 = c[(i+2)*n+j+2];
		for (k = 0; k < n; k ++)
		{
			register double a1 = a[i*n+k];
			register double a2 = a[(i+1)*n+k];
			register double a3 = a[(i+2)*n+k];
			register double b1 = b[k*n+j];
			register double b2 = b[k*n+j+1];
			register double b3 = b[k*n+j+2];
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
		c[i*n+j] = c11;
		c[i*n+j+1] = c12;
		c[i*n+j+2] = c13;
		c[(i+1)*n+j] = c21;
		c[(i+1)*n+j+1] = c22;
		c[(i+1)*n+j+2] = c23;
		c[(i+2)*n+j] = c31;
		c[(i+2)*n+j+1] = c32;
		c[(i+2)*n+j+2] = c33;
	  }
	return 0;
}

double verification(double *c1, double *c2, int n)
{
	double diff = fabs(c1[0] - c2[0]);
	int i;
	for (i = 0; i < n*n; i++)
	  if (fabs(c1[i] - c2[i]) > diff) diff = fabs(c1[i] - c2[i]);
	return diff;
}

int main(int argc, char *argv[])
{
	int n, i;
	double *a, *b, *c0, *c1, *c2, *c3;
	double t0, t1, t2, t3;
	double diff1, diff2, diff3;
	struct timespec start, end;

	n = atoi(argv[1]);
	a = (double *) malloc(n*n*sizeof(double));
	b = (double *) malloc(n*n*sizeof(double));
	c0 = (double *)malloc(n*n*sizeof(double));
	c1 = (double *)malloc(n*n*sizeof(double));
	c2 = (double *)malloc(n*n*sizeof(double));
	c3 = (double *)malloc(n*n*sizeof(double));

	RandMatrixGen(a, n);
	RandMatrixGen(b, n);

	clock_gettime(CLOCK_MONOTONIC, &start);
	dgemm0(a, b, c0, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t0 = (end.tv_sec - start.tv_sec)*BILLION + end.tv_nsec - start.tv_nsec;


	clock_gettime(CLOCK_MONOTONIC, &start);
	dgemm1(a, b, c1, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t1 = (end.tv_sec - start.tv_sec)*BILLION + end.tv_nsec - start.tv_nsec;

	clock_gettime(CLOCK_MONOTONIC, &start);
	dgemm2(a, b, c2, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t2 = (end.tv_sec - start.tv_sec)*BILLION + end.tv_nsec - start.tv_nsec;

	clock_gettime(CLOCK_MONOTONIC, &start);
	dgemm3(a, b, c3, n);
	clock_gettime(CLOCK_MONOTONIC, &end);
	t3 = (end.tv_sec - start.tv_sec)*BILLION + end.tv_nsec - start.tv_nsec;

	diff1 = verification(c0, c1, n);
	diff2 = verification(c0, c2, n);
	diff3 = verification(c0, c3, n);

	printf ("matrix size: %d\n", n);
	printf ("dgemm0 runtime: %llu nanoseconds\n", (long long unsigned int) t0);
	printf ("dgemm1 runtime: %llu nanoseconds\n", (long long unsigned int) t1);
	printf ("dgemm2 runtime: %llu nanoseconds\n", (long long unsigned int) t2);
	printf ("dgemm3 runtime: %llu nanoseconds\n", (long long unsigned int) t3);
	printf ("maximum difference between dgemm0 and dgemm1: %f\n", diff1);
	printf ("maximum difference between dgemm0 and dgemm2: %f\n", diff2);
	printf ("maximum difference between dgemm0 and dgemm3: %f\n", diff3);

	return 0;

}

