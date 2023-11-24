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
gout = np.zeros((100, 2))
t = 0; i = 0; x = init
for i in range(100):
    if (t >= i):
        gout[i,:] = x
    else:
        while(True):
            si = beta*x[0]*x[1]/N
            ir = gamma*x[1]
            h0 = si + ir
            if (h0 < 1e-8):
                t = 1e99
            else:
                t = t + np.random.exponential(1.0/h0)
            if ((t >= i)|(h0 > 1e6)):
                gout[i,:] = x
                break
            if (np.random.random() < si/h0):
                x[0] = x[0] - 1
                x[1] = x[1] + 1
            else:
                x[1] = x[1] - 1

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
# recover beta and gamma parameters based on stochastic observations
data = gout
print(data)

def loss(bg):
    print(bg)
    beta = bg[0]
    gamma = bg[1]
    out = sp.integrate.solve_ivp(sir(beta, gamma),
                (0, 100), init, t_eval=range(100))
    simI = out.y.T
    diff = simI - data
    l = np.sum(diff*diff)
    print(l)
    return l

print(loss([0.3,0.1]))

print("Running optimizer...")
opt = sp.optimize.minimize(loss, [0.2, 0.2],
                           bounds=((0,5),(0,5)))

print(opt)
print("Estimated beta and gamma:")
print(opt.x)


# eof

