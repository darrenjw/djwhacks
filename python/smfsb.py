# smfsb.py
# Start to implement a few smfsb functions in python
# Using numpy for now, but may switch to JAX at some point

import numpy as np
# import scipy as sp

# class for SPN models

class Spn:
    
    def __init__(self, pre, post, h, m):
        self.pre = np.matrix(pre)
        self.post = np.matrix(post)
        self.h = h
        self.m = np.array(m)
        
    def __str__(self):
        return "pre: {}\npost: {}\nh: {}\nm: {}".format(str(self.pre),
                                str(self.post), str(self.h), str(self.m))

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
                if (h0 > 1e06):
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

# TODO: want stepPTS, too

    
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
    
lv = Spn([[1,0],[1,1],[0,1]], [[2,0],[0,2],[0,0]],
         lambda x, t: np.array([x[0], 0.005*x[0]*x[1], 0.6*x[1]]),
         [50,100])

sir = Spn([[1,1,0],[0,1,0]], [[0,2,0],[0,0,1]],
          lambda x, t: np.array([0.1*x[0]*x[1], x[2]]),
          [100,1,0])

# some test code
if __name__ == '__main__':
    stepLv = lv.stepGillespie()
    stepSir = sir.stepGillespie()

    print(lv)
    print(lv.m)
    print(stepLv(lv.m, 0, 1.0))
    print(stepLv(lv.m, 0, 1.0))

    print("First generate a time series")
    out = simTs(lv.m, 0, 100, 0.1, stepLv)

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(2)
    for i in range(2):
        axis[i].plot(range(out.shape[0]), out[:,i])
        axis[i].set_title(f'Time series for variable {i}')
    plt.savefig("lv-ts.png")

    print("Next look at a transition kernel (slow)")
    mat = simSample(1000, lv.m, 0, 10, stepLv)
    figure, axis = plt.subplots(2)
    for i in range(2):
        axis[i].hist(mat[:,i],30)
        axis[i].set_title(f'Histogram for variable {i}')
    plt.savefig("lv-hist.png")

    # TODO: SIR test

    # TODO: PTS test

    
# eof

