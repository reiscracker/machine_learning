#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
import math

logistic_function = lambda x: 1. / ( 1 + np.exp(-x)  )

def logistic_hypothesis(theta):
    return lambda x: 1. / ( 1 + np.exp(- theta.transpose() * x))

x = np.linspace(-5, 5)
plt.figure()
plt.plot(x, logistic_function(x), 'b-')
plt.title("Logistic function")
plt.show()
