/*
 * Sieve of Eratosthenes
 *
 * Programmed by Michael J. Quinn
 *
 * Last modification: 7 September 2001
 */
#include "mpi.h" 
#include <math.h> 
#include <stdio.h>
#include <stdlib.h> 
#define MIN(a,b) ((a)<(b)?(a):(b))

int main (int argc, char *argv[]) {
   unsigned int count; /* Local prime count */
   double elapsed_time; /* Parallel execution time */
   unsigned long  first; /* Index of first multiple */
   unsigned int global_count; /* Global prime count */
   unsigned long long high_value; /* Highest value on this proc */
   unsigned long long i;
   int id; /* Process ID number */
   unsigned long index; /* Index of current prime */
   unsigned long long low_value; /* Lowest value on this proc */
   char *marked; /* Portion of 2,...,'n' */
   unsigned long long int  n; /* Sieving from 2, ..., 'n' */
   int p; /* Number of processes */
   unsigned long proc0_size; /* Size of proc 0's subarray */
   unsigned long prime; /* Current prime */
   unsigned long long size; /* Elements in 'marked' */
   unsigned long long n_size; /* Number of odds between 3 to n */

   MPI_Init (&argc, &argv);
   /* Start the timer */
   MPI_Comm_rank (MPI_COMM_WORLD, &id);
   MPI_Comm_size (MPI_COMM_WORLD, &p);
   MPI_Barrier(MPI_COMM_WORLD);
   elapsed_time = -MPI_Wtime();
   if (argc != 2) {
      if (!id) printf ("Command line: %s <m>\n", argv[0]);
      MPI_Finalize();
      exit (1);
   }
   //n = atoll(argv[1]);
   char *e;
   n = strtoull(argv[1], &e, 10);

   /* Compute number of odds between 3 to n */
   n_size = (n + 1) / 2;

   /* Figure out this process's share of the array, as
      well as the integers represented by the first and
      last array elements */
   low_value  = 2 + (2 * (id * (n_size - 1) / p)) + 1;
   high_value = 2 + (2 * (((id + 1) * (n_size - 1) / p) - 1)) + 1;
   size = (high_value - low_value + 1) / 2;
   /* Bail out if all the primes used for sieving are
      not all held by process 0 */
   proc0_size = (n_size - 1) / p;
   if ((2 + (2 * proc0_size) + 1) < (int) sqrt((double) n)) {
      if (!id) printf ("Too many processes\n");
      MPI_Finalize();
      exit (1);
   }
   /* Allocate this process's share of the array. */
   //printf("ID=%d\tLOW_VALUE=%d\tHIGH_VALUE=%d\tSIZE=%d\n", id, low_value, high_value, size);
   //fflush(stdout);
   marked = (char *) malloc (size);
   if (marked == NULL) {
      printf ("Cannot allocate enough memory\n");
      MPI_Finalize();
      exit (1);
   }
   for (i = 0; i <= size; i++) marked[i] = 0;
   if (!id) index = 0;
   /* Start from 3 */
   prime = 3;
   do {
      if (prime * prime > low_value)
         first = (prime * prime - low_value) / 2;
      else {
         if (!(low_value % prime)) first = 0;
         else{
              first = prime - (low_value % prime);
              if(!(first % 2)) first = first / 2;
              else first = (first + prime) / 2; 
         }
      }
      for (i = first; i <= size; i += prime){
         marked[i] = 1;
      }
      if (!id) {
         while (marked[++index]);
         prime = 2 + (2 * index + 1);
      }
      if (p > 1) MPI_Bcast (&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
   } while (prime * prime <= n);
   count = 0;
   for (i = 0; i <= size; i++)
      if (!marked[i]){
          count++;
      }

   if (p > 1) MPI_Reduce (&count, &global_count, 1, MPI_INT, MPI_SUM,
      0, MPI_COMM_WORLD);
   /* Stop the timer */
   elapsed_time += MPI_Wtime();
   /* Print the results */
   if (!id) {
      global_count++;  //Counting 2 as prime...
      printf("S1, %llu, %d, %lu, %10.6f\n", n, p, global_count, elapsed_time);
   }
   MPI_Finalize ();
   return 0;
}
