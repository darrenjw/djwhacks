#!/usr/bin/env python3
# eigs.py

import numpy as np
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import os

if (not os.path.isfile("zip.train.gz")):
    print("Downloading data from web")
    urllib.request.urlretrieve("https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz", "zip.train.gz")
    
zip_train = pd.read_csv("zip.train.gz", header=None, sep='\s+')
print(zip_train.shape)

digit = zip_train.iloc[:,0].values
print(digit.shape)

images = zip_train.iloc[:,1:].to_numpy()
print(images.shape)

mean = np.mean(images, axis=0)
centred_images = images - mean # works via broadcasting

u, d, vt = np.linalg.svd(centred_images, full_matrices=False)
print(u.shape)
print(d.shape)
print(vt.shape)
v = vt.T

fig, ax = plt.subplots(2,2)
ax[0, 0].imshow(v[:,0].reshape(16, 16))
ax[0, 0].set_title("Eigenvector 1")
ax[0, 1].imshow(v[:,1].reshape(16, 16))
ax[0, 1].set_title("Eigenvector 2")
ax[1, 0].imshow(v[:,2].reshape(16, 16))
ax[1, 0].set_title("Eigenvector 3")
ax[1, 1].imshow(v[:,3].reshape(16, 16))
ax[1, 1].set_title("Eigenvector 4")
plt.show()

# eof
