# sir.py
# Stochastic and deterministic SIR model, and fitting to data

import scipy as sp
import numpy as np

N = 200
beta = 0.3
gamma = 0.1
i0 = 3 # initial number of infected

# Don't bother with R in state since it can be recovered from
# the conservation equation R = N - S - I

init = np.array([N - i0, i0])

# Do a stochastic simulation using the Gillespie algorithm

# TODO: replace with elementary code?!
import smfsb
step = smfsb.sir.stepGillespie()
gout = smfsb.simTs(smfsb.sir.m, 0, 99, 1, step)



# Plot the stochastic realisation
import matplotlib.pyplot as plt
figure, axis = plt.subplots(2)
for i in range(2):
    axis[i].plot(range(100), gout[:,i])
    axis[i].set_title(f'Time series for SIR variable {i}')
plt.savefig("sir-ts-stoch.png")

# Now do a deterministic model
def sir(beta, gamma):
    def rhs(t, si):
        S = si[0]
        I = si[1]
        return np.array([-beta*S*I/N, beta*S*I/N - gamma*I])
    return rhs

rhs = sir(beta, gamma)

# Numerically integrate the SIR ODE
out = sp.integrate.solve_ivp(rhs, (0, 100), init, t_eval=range(100))

print(out)

# Plot the ODE solution
import matplotlib.pyplot as plt
figure, axis = plt.subplots(2)
for i in range(2):
    axis[i].plot(out.t, out.y[i,:])
    axis[i].set_title(f'Time series for SIR variable {i}')
plt.savefig("sir-ts-ode.png")

# Now pretend that we don't know the paramters and attempt to
# recover beta and gamma parameters based on observations of I only
#data = out.y[1,:]
data = gout[:,1]

def loss(bg):
    beta = bg[0]
    gamma = bg[1]
    out = sp.integrate.solve_ivp(sir(beta, gamma),
                (0, 100), init, t_eval=range(100))
    simI = out.y[1,:]
    diff = simI - data
    return np.sum(diff*diff)

print("Running optimizer...")
opt = sp.optimize.minimize(loss, [1, 1],
            method='Nelder-Mead', bounds = ((0, None), (0, None)))

print(opt)
print("Estimated beta and gamma:")
print(opt.x)


# eof

