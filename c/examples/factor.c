/*
factor.c

Factor a number, by simple trial division

 */

#include <stdlib.h>
#include <stdio.h>

void factor(long);

int main(int argc, char *argv[]) {
  long n;
  char *junk;
  if (argc != 2) {
    printf("Usage: factor <number>\n");
    exit(1);
  }
  printf("Parsing input...\n");
  n = strtol(argv[1], &junk, 0);
  printf("Read: %ld.\n", n);
  factor(n);
  exit(0);
}


void factor(long n) {
  long i;
  for (i=2; i<n; i++) {
    if (n % i == 0) {
      printf("%ld x ", i);
      factor(n / i);
      return;
    }
  }
  printf("%ld\n", n);
}


    
/* eof */

