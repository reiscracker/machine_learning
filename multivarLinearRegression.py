#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt

separator = 30*"-"+"\n"

m = dataSets = 10
n = features = 2
min_x = -10
max_x = 10

# Create x values
x = np.random.uniform(min_x, max_x, (dataSets, features))
# Join the x0 column
x0Column = np.ones(( m, 1))
x = np.concatenate( (x0Column, x), axis=1)

# def linear_hypothesis(theta0, theta1, theta2):
#     def costFunc(xVector):
#         print("Got xVector: ")
#         print(xVector)
#         print("(shape " + str(xVector.shape))
#         return theta0 + theta1 * xVector[0] + theta2 * xVector[1]
#     return costFunc

def vector_linear_hypothesis(thetaVector):
    def costFunc(xVector):
        print("Got xVector of shape %s and have theta of shape %s " % (str(xVector.shape), str(thetaVector.shape)))
        result = np.dot(xVector.transpose(), thetaVector)
        return result
    return costFunc

thetas = np.array([1.1, 2.0, -.9]) 
h = vector_linear_hypothesis(thetas)
print("Thetas shape: %s , x shape: %s " % (thetas.shape, x.shape))

# Generate y values using the linear hypothesis function
y = []
for dataSet in x:
    print("Calculating %s" % str(dataSet))
    y.append(h(dataSet))
# And store in a numpy array
y = np.array(y)
print("Y: " + separator)
print(y)

