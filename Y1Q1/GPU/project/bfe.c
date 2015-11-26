#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <libpq-fe.h>

int main(){
    const int TIMESTAMP = 1;
    const int EPSILON = 200;

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
	int M = 0; int m;
	int N = 0; int n;
    while (fgets(line, 1024, in)){
		const char* tok;
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
		const char* tok;
        oid = atoi(strtok(line, ";"));
        time = atoi(strtok(NULL, ";\n"));
		if(time != TIMESTAMP) continue;
        lat = atoi(strtok(NULL, ";\n"));
        lon = atoi(strtok(NULL, ";\n"));
        grid_id = ((lat - min_lat) / EPSILON) * M + ((lon - min_lon) / EPSILON);

        fprintf(out, "%d;%hi;%d;%d;%li\n", oid, time, lat, lon, grid_id);
	}
	printf("Number of points:\t%d\n", i);
	printf("M x N : %d x %d", M, N);
}
