#include <string.h>
#include <stdlib.h>
#include "shapefil.h"

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