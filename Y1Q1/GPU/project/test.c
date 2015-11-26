/*
 *  libpq sample program
 */

#include <stdio.h>
#include <stdlib.h>
#include "libpq-fe.h"   /* libpq header file */

int main() {
    char state_code[3];  /* holds state code entered by user */
    char query_string[256];      /* holds constructed SQL query */
    PGconn     *conn;   /* holds database connection */
    PGresult   *res;    /* holds query result */
    int i;

    conn = PQconnectdb("dbname=trajectories user=and password=nancy port=5432 host=localhost");  /* connect to the database */

    if (PQstatus(conn) == CONNECTION_BAD)       /* did the database connection fail? */
    {
        fprintf(stderr, "Connection to database failed.\n");
        fprintf(stderr, "%s", PQerrorMessage(conn));
        exit(1);
    }

    printf("Enter a state code:  ");    /* prompt user for a state code */
    scanf("%2s", state_code);

    sprintf(query_string,       /* create an SQL query string */
    "SELECT * \
     FROM oldenburg \
     WHERE otime = '%s'", state_code);

    res = PQexec(conn, query_string);   /* send the query */

    if (PQresultStatus(res) != PGRES_TUPLES_OK) /* did the query fail? */
    {
        fprintf(stderr, "SELECT query failed.\n");
        PQclear(res);
        PQfinish(conn);
        exit(1);
    }

    for (i = 0; i < PQntuples(res); i++){/* loop through all rows returned */
        printf("%s\t", PQgetvalue(res, i, 0));  /* print the value returned */
        printf("%s\t", PQgetvalue(res, i, 1));  /* print the value returned */
        printf("%s\t", PQgetvalue(res, i, 2));  /* print the value returned */
        printf("%s\n", PQgetvalue(res, i, 3));  /* print the value returned */

    }
    PQclear(res);       /* free result */

    PQfinish(conn);     /* disconnect from the database */

    return 0;
}
