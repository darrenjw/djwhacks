## dct.R

## Some initial tests using R, checking against existing implementations
x = c(1,2,3,2,4,3)
print("Input")
print(x)

## Naive DFT implementation ( O(n^2) ) for test/reference
dft <- function(z, inverse=FALSE) {
       n <- length(z)
       if(n == 0) return(z)
       k <- 0:(n-1)
       ff <- (if(inverse) 1 else -1) * 2*pi * 1i * k/n
       vapply(1:n, function(h) sum(z * exp(ff*(h-1))), complex(1))
     }

print("dft(x)")
print(dft(x))

## Compare against a real FFT
print("fft(x)")
print(fft(x))

## Inverse DFT
print("iDFT")
print(dft(x,TRUE))
      
## Compare against inverse FFT
print("iFFT")
print(fft(x,TRUE))

## Check it inverts
print("Check it inverts")
print(Re(dft(dft(x),TRUE)/length(x)))

## DCT (DCT-II)
mydct = function(x) {
    N = length(x)
    y = c(x, x[N:1])
    Y = dft(y)
    k = 0:(2*N-1)
    sY = Y * exp(-pi*1i*k/(2*N)) / 2
    Re(sY[1:N])
}

print("My DCT")
print(mydct(x))

## Compare against package implementation
library(dtt)
print("library DCT")
print(dct(x))

## Inverse DCT (DCT-III)

idct = function(x) {
    N = length(x)
    y = c(x, 0, -x[N:2])
    k = 0:(2*N-1)
    sy = y * exp(pi*1i*k/(2*N))
    Y = dft(sy, TRUE)
    Re(Y[1:N])/N
}

print("iDCT")
print(idct(x))

## Compare against library
print("library inverse DCT")
print(dct(x,inverted=TRUE))

## Check it inverts

print("Check it inverts")
print(idct(mydct(x)))

## 2d Inverse DCT
idct2 = function(X) {
    for (i in 1:nrow(X))
        X[i,] = idct(X[i,])
    for (j in 1:ncol(X))
        X[,j] = idct(X[,j])
    X
}

X = matrix(c(1,2,2,2,3,2),ncol=3)
print(X)
print("idct2")
print(idct2(X))

## Library version
print("mvdct")
print(mvdct(X, inverted=TRUE))

## eof

