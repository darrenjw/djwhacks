# sir2.py
# Stochastic and deterministic SIR model, and fitting to data

import scipy as sp
import numpy as np

N = 200
# N = 800 # better estimation with bigger population size (less stochastic)
beta = 0.3
gamma = 0.1
i0 = 3 # initial number of infected

# Don't bother with R in state since it can be recovered from
# the conservation equation R = N - S - I

init = np.array([N - i0, i0])

# Do a stochastic simulation using the Gillespie algorithm
# Record time and state for every event
gstates = np.zeros((2*N, 2)) # 2N is an upper bound on number of events
gtimes = np.zeros(2*N)
t = 0; i = 0; x = init.copy()
gstates[i,:] = x
while (x[1] > 0):
    i = i + 1
    si = beta*x[0]*x[1]/N
    ir = gamma*x[1]
    h0 = si + ir
    if (h0 < 1e-8):
        t = 1e99
    else:
        t = t + np.random.exponential(1.0/h0)
    if (np.random.random() < si/h0):
        x[0] = x[0] - 1
        x[1] = x[1] + 1
    else:
        x[1] = x[1] - 1
    gtimes[i] = t
    gstates[i,:] = x
gtimes = gtimes[range(i+1)]
gstates = gstates[range(i+1),:]
    
# Plot the stochastic realisation
import matplotlib.pyplot as plt
figure, axis = plt.subplots(2)
for i in range(2):
    axis[i].plot(gtimes, gstates[:,i])
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

# This is all a bit dumb, since if we have perfect observations we can
# exactly compute the MLEs for the parameters for the original stochastic model...

data = gstates.T

# squared error loss at observation times
# note that this is not a (good) estimate of the l2 distance between the curves
# since observations are not uniformly distributed in time
def loss(bg):
    print(bg)
    beta = bg[0]
    gamma = bg[1]
    out = sp.integrate.solve_ivp(sir(beta, gamma),
                    (0, gtimes[-1]), init, t_eval=gtimes)
    diff = out.y - data
    l = np.sum(diff*diff)
    print(l)
    return l

print(loss([0.3,0.1]))

print("Running optimizer...")
opt = sp.optimize.minimize(loss, [1, 1],
                           bounds=((0,5),(0,5)))

print(opt)
print("Estimated beta and gamma:")
print(opt.x)


# eof

