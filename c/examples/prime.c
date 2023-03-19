/*
prime.c

Determine whether a number is prime, by simple trial division

 */

#include <stdlib.h>
#include <stdio.h>

int isPrime(long);

int main(int argc, char *argv[]) {
  long n;
  char *junk;
  if (argc != 2) {
    printf("Usage: prime <number>\n");
    exit(1);
  }
  printf("Parsing input...\n");
  n = strtol(argv[1], &junk, 0);
  printf("Read: %ld.\n", n);
  if (isPrime(n)) {
    printf("Prime\n");
    } else {
    printf("Not prime\n");
  }
  exit(0);
}


int isPrime(long n) {
  long i;
  for (i=2; i<n; i++) {
    if (n % i == 0) {
      return(0);
    }
  }
  return(1);
}


    
/* eof */

