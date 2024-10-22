## countdown.R

library(words)
numLetters = 9

print("Countdown game")

vowels = c("a", "e", "i", "o", "u")
consts = setdiff(letters, vowels)
myLetters = NULL
print("Enter v for vowel and c for constanant")
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

print("Your letters are now chosen")
print(myLetters)
print(paste(myLetters, collapse=""))

print("Hit return to start search...")
readline()

counts = tapply(c(rep(1, numLetters), rep(0, 26)), c(myLetters, letters), sum)

for (word in words$word) {
    lets = strsplit(word, "")[[1]]
    lw = length(lets)
    wc = tapply(c(rep(1, lw), rep(0, 26)), c(lets, letters), sum)
    if (all(wc <= counts))
        print(word)
    }

## eof

