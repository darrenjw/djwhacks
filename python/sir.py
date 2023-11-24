# sir.py
# Deterministic SIR model, and fitting to data

import scipy as sp
import numpy as np

N = 200
beta = 0.1
gamma = 0.2

init = np.array([N, 1])

def rhs(t, si):
    S = si[0]
    I = si[1]
    return np.array([-beta*S*I/N, beta*S*I/N - gamma*I])

print(rhs(0, init))

out = sp.integrate.solve_ivp(rhs, (0, 100), init, t_eval=range(100))

print(out)

# eof

