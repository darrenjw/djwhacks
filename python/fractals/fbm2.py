#!/usr/bin/env python3

# Diamond square algorithm...


import numpy as np
import matplotlib.pyplot as plt

# Choose N and H:
N = 5
H = 0.95


# Derived quantities
n = 2**N + 1
D = 3 - H

print(f'N={N}, n={n}, H={H}, D={D}')
if (N>6):
    print("***Large N*** - will take a while!")

mat = np.zeros([n, n])
# Accessor function which wraps around (for the diamond step)
def m(r,c):
    if (r < 0):
        r = r + n
    if (c < 0):
        c = c + n
    if (r >= n):
        r = r - n
    if (c >= n):
        c = c - n
    r = int(r)
    c = int(c)
    return(mat[r,c])

# TODO: Initialise corners if you don't want zeros

def squareFill(L, tlr, tlc, brr, brc):
    #print(f'squareFill({L},{tlr},{tlc},{brr},{brc})')
    shift = (brr - tlr)/2
    cr = tlr + shift
    cc = tlc + shift
    if ((cr < 0)|(cr >= n)|(cc < 0)|(cc >= n)):
        return
    mean = 0.25*(m(tlr,tlc)+m(tlr,brc)+m(brr,tlc)+m(brr,brc))
    mat[int(cr),int(cc)] = mean + np.random.normal()*2**(-H*(N-L))
    if (L > 0):
        diamondFill(L, tlr, tlc-shift, brr, brc-shift)
        diamondFill(L, tlr, tlc+shift, brr, brc+shift)
        diamondFill(L, tlr-shift, tlc, brr-shift, brc)
        diamondFill(L, tlr+shift, tlc, brr+shift, brc)


def diamondFill(L, tlr, tlc, brr, brc):
    #print(f'diamondFill({L},{tlr},{tlc},{brr},{brc})')
    shift = (brr - tlr)/2
    cr = tlr + shift
    cc = tlc + shift
    mean = 0.25*(m(tlr,cc)+m(cr,tlc)+m(cr,brc)+m(brr,cc))
    mat[int(cr),int(cc)] = mean + np.random.normal()*2**(-H*(N-L))
    if (L > 1):
        squareFill(L-1, tlr, tlc, tlr+shift, tlc+shift)
        squareFill(L-1, tlr, tlc+shift, tlr+shift, brc)
        squareFill(L-1, tlr+shift, tlc, brr, tlc+shift)
        squareFill(L-1, tlr+shift, tlc+shift, brr, brc)

print("Filling now...")        
squareFill(N, 0, 0, n-1, n-1)
print("Matrix filled. Rendering...")

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(range(mat.shape[0]), range(mat.shape[1]))
ax.plot_surface(x, y, mat)
plt.title(f'2d fBm via diamond-square: H={H}, D={D}')
plt.show()
