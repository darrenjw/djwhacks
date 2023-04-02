/*
statsv.c

Compute the mean and standard deviation of some numbers

The number of numbers should be provided as a command line argument

The actual numbers should be entered one per line or piped in

eg. ./statsv 5 < fiveNumbers.txt

Requires math library: compile with "-lm". eg. "gcc statsv.c -lm"

 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
  int len;
  double *v;
} vector;

vector * vector_alloc(long);
void vector_free(vector *);
double vector_get(vector *, long);
void vector_set(vector *, long, double);
long vector_len(vector *);

double mean(vector *);
double stddev(vector *, double);


int main(int argc, char *argv[]) {
  double mn, sd;
  vector *x;
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
  x = vector_alloc(n);
  line = NULL;
  for (i=0; i<n; i++) {
    getline(&line, &len, stdin);
    vector_set(x, i, strtod(line, &junk));
  }
  free(line);
  mn = mean(x);
  sd = stddev(x, mn);
  printf("Mean is: %f and stdev is %f.\n", mn, sd);
  exit(0);
}

// vector functions

vector * vector_alloc(long n) {
  vector *x;
  x = malloc(sizeof(vector));
  //(*x).len = n; // could use this - but arrow notation nicer
  x->len = n;
  x->v = calloc(n, sizeof(double));
  return(x);
}

void vector_free(vector * x) {
  free(x->v);
  free(x);
}

double vector_get(vector * x, long i) {
  if ((i < 0)|(i >= vector_len(x))) {
    perror("vector_get index out of bounds\n");
    exit(EXIT_FAILURE);
  }
  return(x->v[i]);
}

void vector_set(vector * x, long i, double xi) {
  if ((i < 0)|(i >= vector_len(x))) {
    perror("vector_set index out of bounds\n");
    exit(EXIT_FAILURE);
  }
  x->v[i] = xi;
}

long vector_len(vector * x) {
  return (x->len);
}


// mean and stdev functions

double mean(vector *x) {
  long i, n;
  double s;
  n = vector_len(x);
  s = 0.0;
  for (i=0; i<n; i++) {
    s += vector_get(x,i);
  }
  return(s/n);
}

double stddev(vector *x, double m) {
  long i, n;
  double ss, xi;
  n = vector_len(x);
  ss = 0.0;
  for (i=0; i<n; i++) {
    xi = vector_get(x,i);
    ss += (xi - m)*(xi-m);
  }
  return(sqrt(ss/(n-1)));
}




    
/* eof */

