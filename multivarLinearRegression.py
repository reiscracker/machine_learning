#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb

def printDebug(debugDict, caller=None):
    print("Debug: " + 30*"-")
    if caller: print(caller)
    print("Got debugDict: " + str(len(debugDict)))
    for name, element in debugDict.iteritems():
        print("%s of %s and shape %s: \n %s" % (name, type(element), str(element.shape) if hasattr(element, "shape") else "None", str(element)) )

def scatterPlot( xMatrix, y ):
#     n = len(xMatrix[:,1])
#     if ( len(xMatrix[:,2]) != len(y[:,1]) != n ):
#         print("Invalid dimensions of operand!")
#     if ( xMatrix.shape[0] != y.shape[0] ): 
#         print("Invalid dimensions of operand!")
    assert Matrix.shape[0] != y.shape[0] ): 
        raise(("Invalid dimensions of operand!")
    fig = plt.figure(figsize=(11, 8), dpi=160)
    ax = fig.add_subplot(111, projection='3d', label="Zufaellige Datenmatrix mit kuenstlich erzeugenten Y-Werten")
    ax.scatter(xMatrix[:, 1], xMatrix[:, 2], y, "b", s=60, marker="*")
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    plt.show()

separator = 30*"-"+"\n"

### Aufgabe 1
dataSets = 100
features = 2
min_x = -10
max_x = 10
y_noise_intensity = 2
iterations = 100
alpha = 0.01
original_thetas = thetas = np.array( [1.1, 2.0, -.9] )
costs = {}

# Create x values
xMatrix = np.random.uniform(min_x, max_x, (dataSets, features))
# Join the x0 column
x0Column = np.ones(( dataSets, 1))
xMatrix = np.concatenate( (x0Column, xMatrix), axis=1)
printDebug( {"x matrix" : xMatrix}, "X values created" )


###  Aufgabe 2
def vector_linear_hypothesis(thetaVector):
    def costFunc(xFeatureVector):
        printDebug( {"Theta" : thetaVector, "Feature vector": xFeatureVector}, "In linear hypothesis" )
        return np.dot(xFeatureVector, thetaVector)
    return costFunc

h = vector_linear_hypothesis(thetas)
printDebug( {"Exakte Thetas" : original_thetas} )

###  Aufgabe 3
# Generate y values using the linear hypothesis function
y = np.array( [ h(featureVector) for featureVector in xMatrix ] )
printDebug( {"Y before noise" : y })
# Add some noise to it
y += np.random.randn(dataSets) * y_noise_intensity
printDebug( {"Y noisy" : y })

# Plot as a scatter plot
scatterPlot(xMatrix, y)

def cost_function(xMatrix, y):
    m = len(xMatrix[:,1])
    def squared_error_cost(thetaVector):
        loss = vector_linear_hypothesis(thetaVector)(xMatrix) - y
        printDebug( {"x matrix": xMatrix, "y values" : y, "loss" : loss}, "In cost function" )
        return 1. / ( 2. * m ) * ( loss ** 2 ).sum()
    return squared_error_cost

# j = cost_function(xMatrix, y)

# Merke: theta.transpose() * xi == h_theta(x)
def compute_new_theta(xMatrix, y, thetas, alpha):
    m = len(xMatrix[:,1])
    leSum = xMatrix.transpose().dot( vector_linear_hypothesis(thetas)(xMatrix) - y )
    thetas_neu = thetas - alpha * (1. / m) * leSum
    printDebug( {"New theta" : thetas_neu}, "In compute new theta" )
    return thetas_neu

# Do 10.000 iterations
for i in range(iterations):
    thetas = compute_new_theta(xMatrix, y, thetas, alpha)
    costs[i] = cost_function(xMatrix, y)(thetas)

print(separator)
print("I assume:")
print("thetas: " + str(thetas))
print("(Correct was " + str(original_thetas) )
print(separator)

