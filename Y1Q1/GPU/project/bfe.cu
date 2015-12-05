#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "bfe.h"

int main(){
	const int TIMESTAMP = 1;
    	const int EPSILON = 2000;
	
	FILE *in;
    	FILE *out;
    	in = fopen("oldenburg.csv", "r");
    	out = fopen("output.csv", "w");
    	fprintf(out, "oid;time;lat;lon;grid_id\n");
	char line[1024];
	int n = 0;
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
        	n++;
	}
	int x[n];
	int y[n];
	int g[n];
	int i[n];
	printf("Min and max latitude:\t(%d, %d)\n", min_lat, max_lat);
	printf("Min and max longitude:\t(%d, %d)\n", min_lon, max_lon);
	M = (max_lat - min_lat) / EPSILON + 1;
	N = (max_lon - min_lon) / EPSILON + 1;
	rewind(in);
	int j = 0;
    	while (fgets(line, 1024, in)){
        	oid = atoi(strtok(line, ";"));
        	time = atoi(strtok(NULL, ";\n"));
		if(time != TIMESTAMP) continue;
		lat = atoi(strtok(NULL, ";\n"));
        	lon = atoi(strtok(NULL, ";\n"));
        	g[j] = M * ((N - 1) - ((lon - min_lon) / EPSILON)) + ((lat - min_lat) / EPSILON);
		x[j] = lat;
		y[j] = lon;
		i[j] = j; 
		j++;
        	//fprintf(out, "%d;%hi;%d;%d;%li\n", oid, time, lat, lon, grid_id);
	}
	printf("Number of points:\t%d\n", n);
	printf("M x N : %d x %d\n", M, N);
	//int r = createGrid("grid.shp", EPSILON, min_lat, max_lat, min_lon, max_lon);
	
	thrust::device_vector<int> d_x(x, x + n);
	thrust::device_vector<int> d_y(y, y + n);
	thrust::device_vector<int> d_g(g, g + n);
	thrust::device_vector<int> d_i(i, i + n);
	thrust::sort_by_key(d_g.begin(), d_g.end(), d_i.begin());
	thrust::gather(d_i.begin(), d_i.end(), d_x.begin(), d_x.begin());
	thrust::gather(d_i.begin(), d_i.end(), d_y.begin(), d_y.begin());
   	
	for(j = 0; j < n; j++)
		std::cout << g[j] << "-" << i[j] << "(" << x[j] << "," << y[j] << ")";
	std::cout << std::endl;
	std::cout << std::endl;
	thrust::copy(d_g.begin(), d_g.end(), g);
	thrust::copy(d_i.begin(), d_i.end(), i);
	thrust::copy(d_x.begin(), d_x.end(), x);
	thrust::copy(d_y.begin(), d_y.end(), y);
	for(j = 0; j < n; j++)
		std::cout << g[j] << "-" << i[j] << "(" << x[j] << "," << y[j] << ")";
	std::cout << std::endl;
	std::cout << std::endl;
	//thrust::copy(d_x.begin(), d_x.end(), std::ostream_iterator<int>(std::cout, ","));
	//std::cout << std::endl;
	return 0;
}
