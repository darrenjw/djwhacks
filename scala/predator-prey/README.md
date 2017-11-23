# Predator Prey

## Small app to do PMMH for a noisy predator prey model

10k iters with 1k particles a thin of 20 and a tuning param of 0.1:

```bash
sbt "run 10000 1000 20 0.1"
```

Analyse the results with:

```bash
Rscript analysis.R
```

#### eof
