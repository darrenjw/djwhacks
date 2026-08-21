##Reaction-Diffusion System
##Turing Patterns

#Schnakenberg


import numpy as np
import matplotlib.pyplot as plt



#Grid Size
#size = 100
size = 250
#Spatial Resolution
#dx = 1/size
#dx = 0.5
dx = 0.25


#Time Step
#dt = 0.0005
dt = 0.0001
#Number of Iterations
#steps = 200000
steps = 2500000
#steps = 50000


#Rates of Production
a = 0.01
b = 2


#ICs

m = size
n = size

matu = np.zeros([m, n])
matu = matu + (a + b)

matv = np.zeros([m, n])
matv = matv + (b / ((a+b)*(a+b)))



##Cheetah spots diffusion coeff
#D = 100

#matv[(m//2):(m//2 + 2), (n//2):(n//2 + 2)] = 1


##Jaguar/Leopard rosettes diffusion coeff
#D = 30
D = 20

matu[(m//2):(m//2 + 20), (n//2):(n//2 + 2)] = 1



##Tiger stripes diffusion coeff
#D = 35

#Stripes IC
#matv[(m//2):(m//2 + 100), (n//2):(n//2 + 4)] = 1.5





#Laplacian  (roll for periodic BCs)

def Lap(mat):
    return ((np.roll(mat, 1, 1) + np.roll(mat, -1, 1) + np.roll(mat, 1, 0) + np.roll(mat, -1, 0) - 4*mat) / (dx*dx))

#Matrix for testing
# m = np.arange(12).reshape((3,4))
# np.roll(m, -1, 1)  #rolls to the left so like selecting element k+1



#RHS

def RHSu(matu, matv):
    return (Lap(matu) + a - matu + ((matu) **2)*(matv))

def RHSv(matu, matv):
    return ((D * Lap(matv)) + b - ((matu) **2)*(matv))



#Equations


for t in range(steps):
    print(t)
    matu = matu + (dt * RHSu(matu, matv))
    matv = matv + (dt * RHSv(matu, matv))   #used new matu in the v update which is slightly different to forward euler but maybe slightly better (leapfrog schemes)


## Save some frames in for loop to stitch together for an animation
    
    if t % 10000 == 0:
        i = t//10000
        plt.imsave(f"u{i:05d}.png", matu)
        plt.imsave(f"v{i:05d}.png", matv)



#In ubuntu shell:
#ffmpeg -framerate 25 -i u%05d.png Tigerumovie.mp4
#totem Tigerumovie.mp4




#Plot solution

plt.imshow(matu)
plt.savefig("outputu.pdf")
plt.show()

plt.imshow(matv)
plt.savefig("outputv.pdf")
plt.show()




#Rather than repeat code could put for loop into func which I can then just call???






