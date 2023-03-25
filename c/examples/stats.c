/*
stats.c

Compute the mean and standard deviation of some numbers

The number of numbers should be provided as a command line argument

The actual numbers should be entered one per line or piped in

eg. ./stats 5 < fiveNumbers.txt

Requires math library: compile with "-lm". eg. "gcc stats.c -lm"

 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double mean(double *, long);
double stddev(double *, double, long);

int main(int argc, char *argv[]) {
  double *x, mn, sd;
  long n, i;
  char *junk, *line;
  size_t len;
  if (argc != 2) {
    printf("Usage: stats <N>\n");
    exit(1);
  }
  printf("Parsing input...\n");
  n = strtol(argv[1], &junk, 0);
  printf("Read: %ld.\n", n);
  x = malloc(n * sizeof(double));
  line = NULL;
  for (i=0; i<n; i++) {
    getline(&line, &len, stdin);
    x[i] = strtod(line, &junk);
  }
  free(line);
  mn = mean(x, n);
  sd = stddev(x, mn, n);
  printf("Mean is: %f and stdev is %f.\n", mn, sd);
  exit(0);
}

double mean(double *x, long n) {
  long i;
  double s;
  s = 0.0;
  for (i=0; i<n; i++) {
    s += x[i];
  }
  return(s/n);
}

double stddev(double *x, double m, long n) {
  long i;
  double ss;
  ss = 0.0;
  for (i=0; i<n; i++) {
    ss += (x[i]-m)*(x[i]-m);
  }
  return(sqrt(ss/(n-1)));
}




    
/* eof */

