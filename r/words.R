## words.R

library(words)

longWords = words$word[words$word_length > 7]

secretWord = sample(longWords, 1)
letters = strsplit(secretWord, "")[[1]]
wordLength = length(letters)
scrambledLetters = sample(letters, wordLength)
scrambled = paste0(scrambledLetters, collapse="")

cat("Anagram:", scrambled, "\n")
cat("Hit enter for the answer...")
readline()
cat("The word was:", secretWord, "\n")

## eof

