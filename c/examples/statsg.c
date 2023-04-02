/*
statsg.c

Compute the mean and standard deviation of some numbers

The number of numbers should be provided as a command line argument

The actual numbers should be entered one per line or piped in

eg. ./statsg 5 < fiveNumbers.txt

Requires the GSL library (ubuntu package libgsl-dev)
Compile with:

gcc statsg.c -lm -lgsl -o statsg

 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics_double.h>

int main(int argc, char *argv[]) {
  double mn, sd;
  gsl_vector *x;
  char *junk;
  long n;
  if (argc != 2) {
    printf("Usage: stats <N>\n");
    exit(1);
  }
  printf("Parsing input...\n");
  n = strtol(argv[1], &junk, 0);
  printf("Read: %ld.\n", n);
  x = gsl_vector_alloc(n);
  gsl_vector_fscanf(stdin, x);
  mn = gsl_stats_mean(x->data, x->stride, x->size);
  sd = gsl_stats_sd_m(x->data, x->stride, x->size, mn);
  printf("Mean is: %f and stdev is %f.\n", mn, sd);
  exit(0);
}

    
/* eof */

