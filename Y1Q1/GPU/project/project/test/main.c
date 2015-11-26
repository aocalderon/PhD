/*
 *  libpq sample program
 */

#include <stdio.h>
#include <stdlib.h>
#include "libpq-fe.h"   /* libpq header file */

void error(char *mess){
    fprintf(stderr, "### %s\n", mess);
    exit(1);
}

int main() {
    int min_lat;
    int max_lat;
    int min_lon;
    int max_lon;
    int timestamp = 10;
    PGconn *conn;   /* holds database connection */
    PGresult *res;
    int i;
    char *query;

    conn = PQconnectdb("dbname=trajectories user=and password=nancy port=5432 host=localhost");  /* connect to the database */

    if (PQstatus(conn) == CONNECTION_BAD){       /* did the database connection fail? */
        fprintf(stderr, "Connection to database failed.\n");
        fprintf(stderr, "%s", PQerrorMessage(conn));
        exit(1);
    }

    char *tag = "oldenburg";
    int epsilon = 200, x1 = 11, y1 = 21, x2 = 101, y2 = 201;
    sprintf(query,       /* create an SQL query string */
        "INSERT INTO grids VALUES ('%s', %d, %d, %d, %d, %d);", tag, epsilon, x1, y1, x2, y2);
    printf("%s", query);
    res = PQexec(conn, query);   /* send the query */
    if (PQresultStatus(res) != PGRES_COMMAND_OK)
        error(PQresultErrorMessage(res));

    PQclear(res);
    PQfinish(conn);     /* disconnect from the database */

    return 0;
}

