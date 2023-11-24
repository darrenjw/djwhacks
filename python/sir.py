# sir.py
# Deterministic SIR model, and fitting to data

import scipy as sp
import numpy as np

N = 200
beta = 0.3
gamma = 0.1

init = np.array([N, 1])

def rhs(t, si):
    S = si[0]
    I = si[1]
    return np.array([-beta*S*I/N, beta*S*I/N - gamma*I])

print(rhs(0, init))

out = sp.integrate.solve_ivp(rhs, (0, 100), init, t_eval=range(100))

print(out)
#print(out.t)
#print(out.y)

import matplotlib.pyplot as plt
figure, axis = plt.subplots(2)
for i in range(2):
    axis[i].plot(out.t, out.y[i,:])
    axis[i].set_title(f'Time series for SIR variable {i}')
plt.savefig("sir-ts.png")


# Now attempt to recover beta and gamma parameters
# based on observations of I only





# eof

