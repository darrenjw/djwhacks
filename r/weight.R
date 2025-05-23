#!/usr/bin/env Rscript
## weight.R
## Little script to plot my weight data over time
## Data extracted from an Excel spreadsheet

if (!require("pacman")) install.packages("pacman")
pacman::p_load("readxl", "lubridate")

fileName="~/Dropbox/Shared/DarrenAlisonShared/Weight.xls"

xl = read_excel(fileName)
rowCount = dim(xl)[1]
xl = xl[3:rowCount,]
weekDate = xl[[1]]
weight = as.numeric(xl[[10]])
weight[weight == 0] = NA
bmi = as.numeric(xl[[11]])
bmi[bmi == 0] = NA

plot(weekDate, bmi, type="l", col=4, lwd=2, ylim=c(20, 28),
     ylab="BMI", xlab="Time", main="Darren's BMI over time")
abline(h=25, col=2, lwd=1.5)
abline(h=23, col=3)
abline(h=21, col=5)
abline(v=ymd("2019-03-05", tz="UTC"), col=5) # RIP RJB
abline(v=ymd("2020-03-14", tz="UTC"), col=5) # First Covid lockdown
abline(v=ymd("2022-01-04", tz="UTC"), col=5) # Started at Durham

## eof

