#include <stdio.h>
#include <math.h>

int main(int argc, char ** argv){
	int x1 = 1, y1 = 1;
	int x2 = 1, y2 = 3;
	int epsilon = 4;
	
	float r2 = pow(epsilon / 2, 2);
	printf("r2:%f\n", r2);
	int x = x1 - x2;
	int y = y1 - y2;
	printf("(%d,%d)\n", x, y);
	float d2 = pow(x, 2) + pow(y, 2);
	printf("d2:%f\n", d2);
	if(d2 == 0){
		printf("(%d,%d)\n", x1, y1);
		return 0;
	}
	float e = fabsf(4 * (r2 / d2) - 1);
	printf("e:%f\n", e);
	float r = sqrt(e);
	printf("r:%f\n", r);
	float h1 = ((x + y * r) / 2) + x2;
	float k1 = ((y - x * r) / 2) + y2;
	printf("x + y * r:%f\n", x + y * r);
	printf("y - x * r:%f\n", y - x * r);

	printf("(%f,%f)\n", h1, k1);

    return 0;
}
