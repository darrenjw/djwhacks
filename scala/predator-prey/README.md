# Predator Prey

## Small app to do PMMH for a noisy predator prey model

10k iters with 10k particles a thin of 5 and a tuning param of 10^(-11):

```bash
sbt "run 10000 10000 5 1.0e-11"
```

Analyse the results with:

```bash
Rscript analysis.R
```

#### eof
