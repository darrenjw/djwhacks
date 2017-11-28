# Predator Prey

## Small app to do PMMH for a noisy predator prey model

10k iters with 10k particles a thin of 5 and a tuning param of 10^(-11):
```bash
sbt "run 10000 10000 5 1.0e-11"
```

Analyse the results with:
```bash
Rscript analysis.R LvPmmh.csv
```
Note that passing the name of a gzipped MCMC file is fine.

The model is:

$$dX_t = (\mu X_t + \phi X_t V_t)dt + \sqrt{v_X X_t}\,dW_t$$

$$dV_t = (\delta X_t V_t-mV_t)dt + \sqrt{v_V V_t}\,dW'_t$$

And the observations of $X_t$ and $V_t$ have noise $nv_X$ and $nv_V$, respectively.


So, there are a total of 8 parameters estimated by the model. Note that the current version of the model has the 4 LV parameters unconstrained and the 4 noise paramters constrained to be positive. There is code for running a version of the model where all parameters are constrained to be positive, but running it will requiring hacking the source.

#### eof
