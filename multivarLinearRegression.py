#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_debug_function(debug_enabled):
    def d(debugValues, caller=None):
        debugHeader = "{0} {1} lol {2}".format
        debugElements = "Element {n} of type {t} (Shape: {v.shape}):\n{v}".format
        if debug_enabled:
            print( debugHeader("Debug message", caller if caller else "a", 30*"-") )
            print( "\n".join( [ debugElements(n=name, v=value, t=type(value)) for name, value in debugValues.iteritems() ] ) )
    return d
printDebug = get_debug_function(False)

def scatterPlot( xMatrix, y ):
    if ( xMatrix.shape[0] != y.shape[0] ):
        raise("Invalid dimensions of operand!")

    fig = plt.figure(figsize=(11, 8), dpi=160)
    ax = fig.add_subplot(111, projection='3d', label="Zufaellige Datenmatrix mit kuenstlich erzeugenten Y-Werten")
    ax.scatter(xMatrix[:, 1], xMatrix[:, 2], y, "b", s=60, marker="*")
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    plt.show()

def myPlot( xMatrix, y ):
    if ( xMatrix.shape[0] != y.shape[0] ):
        raise("Invalid dimensions of operand!")

    fig = plt.figure(figsize=(11, 8), dpi=160)
    ax = fig.add_subplot(111, projection='3d', label="Zufaellige Datenmatrix mit kuenstlich erzeugenten Y-Werten")
    ax.scatter(xMatrix[:, 1], xMatrix[:, 2], y, "b", s=60, marker="*")
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    plt.show()

### Aufgabe 1
dataSets = 100
features = 2
min_x = -10
max_x = 10
y_noise_intensity = 2
iterations = 1
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
    m = len(xMatrix)
    def squared_error_cost(thetaVector):
        loss = vector_linear_hypothesis(thetaVector)(xMatrix) - y
        printDebug( {"x matrix": xMatrix, "y values" : y, "loss" : loss}, "In cost function" )
        return 1. / ( 2. * m ) * ( loss ** 2 ).sum()
    return squared_error_cost

j = cost_function(xMatrix, y)

# Merke: theta.transpose() * xi == h_theta(x)
def compute_new_theta(xMatrix, y, thetas, alpha):
    m = len(xMatrix)
    leSum = xMatrix.transpose().dot( vector_linear_hypothesis(thetas)(xMatrix) - y )
    thetas_neu = thetas - alpha * (1. / m) * leSum
    printDebug( {"New theta" : thetas_neu}, "In compute new theta" )
    return thetas_neu

# Do 10.000 iterations
for i in range(iterations):
    thetas = compute_new_theta(xMatrix, y, thetas, alpha)
    costs[i] = cost_function(xMatrix, y)(thetas)

print("{0} I assume, that I have found something close to the original theta.{0}\n \
        I guess it must have been around.. \n {1}".format( (30*"-"), np.vectorize("%.2f".__mod__)(thetas)))

print("(Original thetas were {}".format(original_thetas))

