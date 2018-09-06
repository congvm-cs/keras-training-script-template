import numpy as np


def lrSched(epoch, lr):
    initial_lrate = 0.05
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    if lrate < 0.001:
        lrate = 0.001
    return lrate