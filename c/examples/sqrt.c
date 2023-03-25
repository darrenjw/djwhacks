/*
sqrt.c

Square root a given number using Newton's method.

 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double mySqrt(double);

int main(int argc, char *argv[]) {
  double x, sx;
  char *junk;
  if (argc != 2) {
    printf("Usage: sqrt <number>\n");
    exit(1);
  }
  printf("Parsing input...\n");
  x = strtod(argv[1], &junk);
  printf("Read: %f.\n", x);
  sx = mySqrt(x);
  printf("Square root is: %f.\n", sx);
  exit(0);
}


double mySqrt(double x) {
  double sx;
  sx = x;
  while (fabs(sx*sx - x) > 1.0e-8) {
    sx = (sx + x/sx)/2;
  }
  return(sx);
}


    
/* eof */

