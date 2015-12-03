#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>  
#include <time.h> 

int * getCenterDisk(int p1[], int p2[], int epsilon){
	static int c[2]; 
	int x1 = p1[0], y1 = p1[1];
	int x2 = p2[0], y2 = p2[1];
	float r2 = pow(epsilon / 2, 2);
	int x = x1 - x2;
	int y = y1 - y2;
	float d2 = pow(x, 2) + pow(y, 2);
	if(d2 == 0){
		return p1;
	}
	float e = fabsf(4 * (r2 / d2) - 1);
	float r = sqrt(e);
	c[0] = (int)(((x + y * r) / 2) + x2);
	c[1] = (int)(((y - x * r) / 2) + y2);
	
	return c;
} 

int main(int argc, char ** argv){
	int p1[2];
	int p2[2];
	p1[0] = 10;
	p1[1] = 10;
	p2[0] = 10;
	p2[1] = 10;
	int epsilon = 100;
	int *c = getCenterDisk(p1, p2, epsilon);
	printf("(%d,%d)\n", c[0], c[1]);
	
	FILE *in;
	in = fopen("test.tsv", "r");
	char line[1024];
    while (fgets(line, 1024, in)){
		const char* tok;
        p1[0] = atoi(strtok(line, "\t"));
        p1[1] = atoi(strtok(NULL, "\t\n"));
        p2[0] = atoi(strtok(NULL, "\t\n"));
        p2[1] = atoi(strtok(NULL, "\t\n"));
		int *c = getCenterDisk(p1, p2, epsilon);
		//printf("%d\t%d\n", c[0], c[1]);
	}	
	/*
	FILE *f;
	f = fopen("test.tsv", "w");
	for(int i = 0; i < 100; i++){
		int xr1 = rand() % 100;
		int yr1 = rand() % 100;
		int xr2 = rand() % 100;
		int yr2 = rand() % 100;
		fprintf(f, "%d\t%d\t%d\t%d\n", xr1, yr1, xr2, yr2);
	}
	fclose(f);
    */
    return 0;
}
