#!/usr/bin/env python3
# smfsb.py
# Start to implement a few smfsb functions in python
# Using numpy for now, but may switch to JAX at some point

import numpy as np
# import scipy as sp

# class for SPN models

class Spn:
    
    def __init__(self, n, t, pre, post, h, m):
        self.n = n # species names
        self.t = t # reaction names
        self.pre = np.matrix(pre)
        self.post = np.matrix(post)
        self.h = h # hazard function
        self.m = np.array(m) # initial marking
        
    def __str__(self):
        return "n: {}\n t: {}\npre: {}\npost: {}\nh: {}\nm: {}".format(str(self.n),
                str(self.t), str(self.pre), str(self.post), str(self.h), str(self.m))

    def stepGillespie(self):
        S = (self.post - self.pre).T
        u, v = S.shape
        def step(x0, t0, deltat):
            t = t0
            x = x0
            termt = t0 + deltat
            while(True):
                h = self.h(x, t)
                h0 = h.sum()
                if (h0 > 1e07):
                    print("WARNING: hazard too large - terminating!")
                    return(x)
                if (h0 < 1e-10):
                    t = 1e99
                else:
                    t = t + np.random.exponential(1.0/h0)
                if (t > termt):
                    return(x)
                j = np.random.choice(v, p=h/h0)
                x = np.add(x, S[:,j].A1)
        return step

    def stepPTS(self, dt = 0.01):
        S = (self.post - self.pre).T
        u, v = S.shape
        def step(x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            while(True):
                h = self.h(x, t)
                r = np.random.poisson(h * dt)
                x = np.add(x, S.dot(r).A1)
                t = t + dt
                if (t > termt):
                    return x
        return step
    

    
# some simulation functions

def simTs(x0, t0, tt, dt, stepFun):
    n = int((tt-t0) // dt) + 1
    u = len(x0)
    mat = np.zeros((n, u))
    x = x0
    t = t0
    mat[1,:] = x
    for i in range(n):
        t = t + dt
        x = stepFun(x, t, dt)
        mat[i,:] = x
    return mat

def simSample(n, x0, t0, deltat, stepFun):
    u = len(x0)
    mat = np.zeros((n, u))
    for i in range(n):
        mat[i,:] = stepFun(x0, t0, deltat)
    return mat


# some example SPN models
    
lv = Spn(["Prey", "Predator"], ["Prey rep", "Inter", "Pred death"],
         [[1,0],[1,1],[0,1]], [[2,0],[0,2],[0,0]],
         lambda x, t: np.array([x[0], 0.005*x[0]*x[1], 0.6*x[1]]),
         [50,100])

sir = Spn(["S", "I", "R"], ["S->I", "I->R"], [[1,1,0],[0,1,0]], [[0,2,0],[0,0,1]],
          lambda x, t: np.array([0.3*x[0]*x[1]/200, 0.1*x[1]]),
          [197, 3, 0])

# TODO: a toy genetic toggle switch?


# some test code
if __name__ == '__main__':
    stepLv = lv.stepGillespie()
    stepSir = sir.stepGillespie()

    print(lv)
    print(lv.m)
    print(stepLv(lv.m, 0, 1.0))
    print(stepLv(lv.m, 0, 1.0))

    print("First generate a LV time series")
    out = simTs(lv.m, 0, 100, 0.1, stepLv)

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(2)
    for i in range(2):
        axis[i].plot(range(out.shape[0]), out[:,i])
        axis[i].set_title(f'Time series for {lv.n[i]}')
    plt.savefig("lv-ts.png")

    print("Next look at a LV transition kernel (slow)")
    mat = simSample(100, lv.m, 0, 10, stepLv)
    figure, axis = plt.subplots(2)
    for i in range(2):
        axis[i].hist(mat[:,i],30)
        axis[i].set_title(f'Histogram for {lv.n[i]}')
    plt.savefig("lv-hist.png")

    print("Generate a SIR time series")
    out = simTs(sir.m, 0, 100, 0.1, stepSir)

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(3)
    for i in range(3):
        axis[i].plot(range(out.shape[0]), out[:,i])
        axis[i].set_title(f'Time series for {sir.n[i]}')
    plt.savefig("sir-ts.png")


    stepLvP = lv.stepPTS()
    print("First generate a LV time series using PTS approx")
    out = simTs(lv.m, 0, 100, 0.1, stepLvP)

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(2)
    for i in range(2):
        axis[i].plot(range(out.shape[0]), out[:,i])
        axis[i].set_title(f'Time series for PTS {lv.n[i]}')
    plt.savefig("lv-pts.png")

    
# eof

