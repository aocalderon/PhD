#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "shapefil.h"

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

int createGrid(char *shapefile, int epsilon, int min_lat, int max_lat, int min_lon, int max_lon){
	const int nVMax = 1000;
	const int nVertices = 2;
	const int nShapeType = SHPT_ARC;
	SHPHandle	hSHP;
	SHPObject	*psObject;
	double	*padfX, *padfY, i;
	int rlat, rlon;

	rlat = (max_lat - min_lat) % epsilon;
	max_lat += epsilon - rlat; 
	rlon = (max_lon - min_lon) % epsilon;
	max_lon += epsilon - rlon; 
	
	hSHP = SHPCreate(shapefile, nShapeType );
	if( hSHP == NULL ){
		printf( "Unable to create:%s\n", shapefile);
		exit(3);
	}
	padfX = (double *) malloc(sizeof(double) * nVMax);
	padfY = (double *) malloc(sizeof(double) * nVMax);
    
	for(i = min_lat; i <= max_lat; i += epsilon){
		padfX[0] = i;
		padfX[1] = i;
		padfY[0] = min_lon;
		padfY[1] = max_lon;
		psObject = SHPCreateSimpleObject(nShapeType, nVertices, padfX, padfY, NULL);
		SHPWriteObject( hSHP, -1, psObject );
	}	
	for(i = min_lon; i <= max_lon; i += epsilon){
		padfX[0] = min_lat;
		padfX[1] = max_lat;
		padfY[0] = i;
		padfY[1] = i;
		psObject = SHPCreateSimpleObject(nShapeType, nVertices, padfX, padfY, NULL);
		SHPWriteObject( hSHP, -1, psObject );
	}  
	SHPDestroyObject(psObject);
	SHPClose(hSHP);

	return 0;
}
