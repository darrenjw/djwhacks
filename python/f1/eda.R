## eda.R
## Some basic sanity checking and EDA for the results data from FastF1

## load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load("nanoparquet")

## file to analyse
fileName = "all-results.parquet"

## load up the data file into a data frame
df = read_parquet(fileName)
str(df)

## Some very basic stats
seasons = unique(sort(df[,'Year']))
cat(paste(length(seasons), "seasons:\n"))
print(seasons)

drivers = unique(sort(df[,'DriverId']))
cat(paste(length(drivers), "drivers:\n"))
print(drivers)

teams = unique(sort(df[,'TeamId']))
cat(paste(length(teams), "teams:\n"))
print(teams)


## Some very basic linear models
df$driver = factor(df$DriverId)
df$team = factor(df$TeamId)

##options(contrasts = c("contr.sum", "contr.poly"))
##options(contrasts = c("contr.treatment", "contr.poly"))

mod = lm(Points ~ driver, data=df)
cat("Regress points on driver (top 10):\n")
print(sort(mod$coefficients, decreasing=TRUE)[1:10])

mod = lm(-Position ~ driver, data=df)
cat("Regress -position on driver:\n")
print(sort(mod$coefficients, decreasing=TRUE)[1:10])

mod = lm(Points ~ team, data=df)
cat("Regress points on team:\n")
print(sort(mod$coefficients, decreasing=TRUE)[1:10])

mod = lm(-Position ~ team, data=df)
cat("Regress -position on team:\n")
print(sort(mod$coefficients, decreasing=TRUE)[1:10])

mod = lm(Points ~ driver + team, data=df)
cat("Regress points on driver + team:\n")
print(sort(mod$coefficients, decreasing=TRUE)[1:20])

mod = lm(-Position ~ driver + team, data=df)
cat("Regress -position on driver + team:\n")
print(sort(mod$coefficients, decreasing=TRUE)[1:20])



## eof

