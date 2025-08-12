#!/usr/bin/env python3
# plot_loss.py
# plot the loss function

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_name):
    df = pd.read_csv(file_name)
    it = df['it']
    loss = df['loss']
    return it, loss

def mpl_plot(it, loss, file_name):
    lloss = np.log(loss)
    plt.figure()
    plt.subplot(211)
    plt.plot(it, lloss)
    plt.title("Loss for full run")
    plt.subplot(212)
    l = len(it)
    plt.plot(it[l//2:l], lloss[l//2:l])
    plt.title("Loss for second half of run")
    plt.savefig(file_name)


if __name__ == '__main__':
    in_file = "loss.csv"
    out_file = "loss.pdf"
    if (len(sys.argv) > 1):
        in_file = sys.argv[1]
    if (len(sys.argv) > 2):
        out_file = sys.argv[2]
    it, loss = read_data(in_file)
    mpl_plot(it, loss, out_file)





# eof

