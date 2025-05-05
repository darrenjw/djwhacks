## eda.R
## Some basic sanity checking and EDA for the results data from FastF1

if (!require("pacman")) install.packages("pacman")
pacman::p_load("nanoparquet")

fileName = "all-results.parquet"

df = read_parquet(fileName)
print(dim(df))
str(df)

seasons = unique(sort(df[,'Year']))
cat(paste(length(seasons), "seasons:\n"))
print(seasons)

drivers = unique(sort(df[,'DriverId']))
cat(paste(length(drivers), "drivers:\n"))
print(drivers)



## eof

