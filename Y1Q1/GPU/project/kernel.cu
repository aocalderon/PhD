#include <stdio.h>

__device__ int distance(int x1, int y1, int x2, int y2){
	int dx = x2 - x1;
	int dy = y2 - y1;
	return (dx * dx) + (dy * dy);
}

__device__ int findPosition(const int *a, int k, int b, int top){
	if(b < 0){
		return -1;
	}
	if(b > top){
		return -2;
	}
	for(int i = 0; i < k; i++){
		if(a[i] == b){
			return i;
		}	
	}
	return -3;
}

__global__ void parallelBFE(const int *x, const int *y, int *g, const int *a, const int *b, int n, int k, int M, int N, int E, unsigned long *N_DISKS){
	int t = threadIdx.x;
	//int px[250];
	//int py[250];
	//int h;
	unsigned long j = 0;
	
	// Center-Medium
	int cm = a[t];
	for(int i = b[t]; i < b[t + 1]; i++){
		//px[j] = x[i];
		//py[j] = y[i];
		j++;
	}
	//h = j;

	// Left-Medium
	int lm;
	if(cm % M == 0){
		lm = -1;
	} else {
		lm = findPosition(a, k, cm - 1, M*N);
	}
	if(lm >= 0){
		for(int i = b[lm]; i < b[lm + 1]; i++){
			//px[j] = x[i];
			//py[j] = y[i];
			j++;
		}
	}
	// Right-Medium
	int rm;
	if(cm % M == M - 1){
		rm = -1;
	} else {
		rm = findPosition(a, k, cm + 1, M*N);
	}
	if(rm >= 0){
		for(int i = b[rm]; i < b[rm + 1]; i++){
			//px[j] = x[i];
			//py[j] = y[i];
			j++;
		}
	}
	// Center-Up
	int cu = cm - M;
	cu = findPosition(a, k, cu, M*N);
	if(cu >= 0){
		for(int i = b[cu]; i < b[cu + 1]; i++){
			//px[j] = x[i];
			//py[j] = y[i];
			j++;
		}
	}
	// Left-Up
	int lu;
	if(cm % M == 0){
		lu = -1;
	} else {
		lu = findPosition(a, k, cm - M - 1, M*N);
	}
	if(lu >= 0){
		for(int i = b[lu]; i < b[lu + 1]; i++){
			//px[j] = x[i];
			//py[j] = y[i];
			j++;
		}
	}
	// Right-Up
	int ru;
	if(cm % M == M - 1){
		ru = -1;
	} else {
		ru = findPosition(a, k, cm - M + 1, M*N);
	}
	if(ru >= 0){
		for(int i = b[ru]; i < b[ru + 1]; i++){
			//px[j] = x[i];
			//py[j] = y[i];
			j++;
		}
	}
	// Center-Down
	int cd = cm + M;
	cd = findPosition(a, k, cd, M*N);
	if(cd >= 0){
		for(int i = b[cd]; i < b[cd + 1]; i++){
			//px[j] = x[i];
			//py[j] = y[i];
			j++;
		}
	}
	// Left-Down
	int ld;
	if(cm % M == 0){
		ld = -1;
	} else {
		ld = findPosition(a, k, cm + M - 1, M*N);
	}
	if(ld >= 0){
		for(int i = b[ld]; i < b[ld + 1]; i++){
			//px[j] = x[i];
			//py[j] = y[i];
			j++;
		}
	}
	// Right-Down
	int rd;
	if(cm % M == M - 1){
		rd = -1;
	} else {
		rd = findPosition(a, k, cm + M + 1, M*N);
	}
	if(rd >= 0){
		for(int i = b[rd]; i < b[rd + 1]; i++){
			//px[j] = x[i];
			//py[j] = y[i];
			j++;
		}
	}
	N_DISKS[t] = j;
}
