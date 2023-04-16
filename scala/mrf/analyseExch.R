#!/usr/bin/env Rscript

## analyse.R

if (!require("pacman")) install.packages("pacman")
pacman::p_load("smfsb", "coda")

out = read.csv("exch.csv", header=TRUE)

mcmcSummary(out)

#pairs(out, pch=19, col=2, cex=0.2)

message("ESS:")
ess = effectiveSize(out)
print(ess)
message(paste("Average ESS:", mean(ess)))
message(paste("Min ESS:", min(ess)))


## eof
