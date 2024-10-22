## words.R

library(words)

longWords = words$word[words$word_length > 7]

secretWord = sample(longWords, 1)
letts = strsplit(secretWord, "")[[1]]
wordLength = length(letts)
scrambledLetters = sample(letts, wordLength)
scrambled = paste0(scrambledLetters, collapse="")

cat("Anagram:", scrambled, "\n")
cat("Hit enter for the answer...\n")
readline()
cat("The word was:", secretWord, "\n")

## eof

