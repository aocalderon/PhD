#include <stdio.h>
#include "bfe.h"

struct Point{
	int x;
	int y;
}

int main(){
    const int TIMESTAMP = 1;
    const int EPSILON = 2000;

    FILE *in;
    FILE *out;
    in = fopen("oldenburg.csv", "r");
    out = fopen("output.csv", "w");
    fprintf(out, "oid;time;lat;lon;grid_id\n");
	char line[1024];
	int i = 0;
	long grid_id;
	int oid; short time;
	int lat; int lon;
	int max_lat = INT_MIN; int min_lat = INT_MAX;
	int max_lon = INT_MIN; int min_lon = INT_MAX;
	int M = 0; 
	int N = 0; 
    while (fgets(line, 1024, in)){
        atoi(strtok(line, ";"));
		if(atoi(strtok(NULL, ";\n")) != TIMESTAMP) continue;
        lat = atoi(strtok(NULL, ";\n"));
        if(lat > max_lat) max_lat = lat;
        if(lat < min_lat) min_lat = lat;
        lon = atoi(strtok(NULL, ";\n"));
        if(lon > max_lon) max_lon = lon;
        if(lon < min_lon) min_lon = lon;
        i++;
	}
	printf("Min and max latitude:\t(%d, %d)\n", min_lat, max_lat);
	printf("Min and max longitude:\t(%d, %d)\n", min_lon, max_lon);
	M = (max_lat - min_lat) / EPSILON + 1;
	N = (max_lon - min_lon) / EPSILON + 1;
	rewind(in);
    while (fgets(line, 1024, in)){
        oid = atoi(strtok(line, ";"));
        time = atoi(strtok(NULL, ";\n"));
		if(time != TIMESTAMP) continue;
        lat = atoi(strtok(NULL, ";\n"));
        lon = atoi(strtok(NULL, ";\n"));
        grid_id = M * ((N - 1) - ((lon - min_lon) / EPSILON)) + ((lat - min_lat) / EPSILON) ;

        fprintf(out, "%d;%hi;%d;%d;%li\n", oid, time, lat, lon, grid_id);
	}
	printf("Number of points:\t%d\n", i);
	printf("M x N : %d x %d\n", M, N);
	int r = createGrid("grid.shp", EPSILON, min_lat, max_lat, min_lon, max_lon);
	
	return r;
}
