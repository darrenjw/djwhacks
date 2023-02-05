## genData.R
## Generate some synthetic data for analysis via HMMs

set.seed(42)

## First, a very short sequence for testing purposes
x = rnorm(30)
x[11:20] = x[11:20] + 3
plot(ts(x))

write(x, "short.txt", sep="\n")

## eof

