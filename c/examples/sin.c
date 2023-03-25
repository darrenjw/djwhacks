/*
sin.c

Compute the sin of a number using the trig identity:

sin(x) = 3sin(x/3) - 4sin^3(x/3)

and the fact that for small enough x, sin(x) ~= x.

 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double mySin(double);

int main(int argc, char *argv[]) {
  double x, sx;
  char *junk;
  if (argc != 2) {
    printf("Usage: sin <number>\n");
    exit(1);
  }
  printf("Parsing input...\n");
  x = strtod(argv[1], &junk);
  printf("Read: %f.\n", x);
  sx = mySin(x);
  printf("Sin is: %f.\n", sx);
  exit(0);
}


double mySin(double x) {
  double sx3;
  if (fabs(x) < 1.0e-5)
    return(x);
  else {
    sx3 = mySin(x/3);
    return(3*sx3 - 4*sx3*sx3*sx3);
  }
}


    
/* eof */

