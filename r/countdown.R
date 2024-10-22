## countdown.R

library(words)

numLetters = 10
myLetters = sample(letters, numLetters, replace=TRUE) # not enough vowels!!!
print(myLetters)

print("Hit return to start search...")
readline()

df = data.frame(l=c(myLetters, letters), c=c(rep(1, numLetters), rep(0, 26)))
counts = tapply(df$c, df$l, sum)

for (word in words$word) {
    lets = strsplit(word, "")[[1]]
    lw = length(lets)
    wc = tapply(c(rep(1, lw), rep(0, 26)), c(lets, letters), sum)
    if (all(wc <= counts))
        print(word)
    }

## eof

