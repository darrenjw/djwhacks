## genData.R
## Generate some synthetic data for analysis via HMMs

set.seed(42)

## First, a very short sequence for testing purposes
x = rnorm(30)
x[11:20] = x[11:20] + 3
plot(ts(x))
write(x, "short.txt", sep="\n")

## Now a bigger sequence
x = rnorm(3000)
x[1001:2000] = x[1001:2000] + 1
plot(ts(x))
write(x, "medium.txt", sep="\n")



## Now a large sequence for stress-testing
x = rnorm(300000)
x[100001:200000] = x[100001:200000] + 0.1
plot(ts(x))
write(x, "large.txt", sep="\n")




## eof

