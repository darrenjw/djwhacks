import random,math

def gibbs(N=50000,thin=100):
    x=0
    y=0
    print "Iter  x  y"
    for i in range(N):
        for j in range(thin):
            x=random.gammavariate(3,1.0/(y*y+4))
            y=random.gauss(1.0/(x+1),1.0/math.sqrt(2*x+2))
        print i,x,y

gibbs()

