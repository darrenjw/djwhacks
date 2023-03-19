/*
e.c

Compute e via its series expansion:

e = 1/0! + 1/1! + 1/2! + 1/3! + ...

*/

#include <stdlib.h>
#include <stdio.h>

int main() {
  int i;
  long denom;
  double e;
  e = 1.0;
  denom = 1;
  for (i=1; i<20; i++) {
    denom *= i;
    e += 1.0/denom;
    printf("%d: %f\n", i, e);
  }
}


/* eof */

