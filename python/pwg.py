#!/usr/bin/env python

import sys,random

def pwgen(n):
  #print "pwgen"+str(n)
  for i in range(n):
    ri=random.randint(0,35)
    if ri<10:
      sys.stdout.write(str(ri))
    else:
      sys.stdout.write(chr(ri+87))
  print


if __name__=='__main__':
  c=len(sys.argv)
  if (c==1):
    pwgen(12)
  else:
    pwgen(eval(sys.argv[1]))
  exit(0)

# eof


