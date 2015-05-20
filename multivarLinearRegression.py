#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt

m = dataSets = 10
n = features = 2
min_x = -10
max_x = 10

# Create x values
x = np.random.uniform(min_x, max_x, (dataSets, features))

def linear_hypothesis(thetas):
    return lambda xVector: thetas[0] + ( x[1:] * thetas ).sum() 

theta = np.array([1.1, 2.0, -.9]) 



