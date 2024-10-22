## countdown.R

library(words)
numLetters = 9

vowels = c("a", "e", "i", "o", "u")
consts = setdiff(letters, vowels)
myLetters = NULL
for (i in 1:numLetters) {
    print("v/c?")
    line = readline()
    if (line == "v")
        let = sample(vowels, 1)
    else
        let = sample(consts, 1)
    myLetters = c(myLetters, let)
    print(myLetters)
    }

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

