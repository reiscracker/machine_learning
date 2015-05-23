#!/usr/bin/python

import numpy as np
import plotty, debug_printer
import sys

isDebug = (len(sys.argv) > 1 and "-d" in sys.argv)
debug = debug_printer.get_debug_function(isDebug)

dataSets, features = 100,  2
min_x, max_x = -10, 10
original_thetas = thetas = np.array( [1, 2, 3] )
y_noise_intensity = 2
iterations = 10
alpha = 0.00001
costs = {}

### Aufgabe 1
# Create x values
xMatrix = np.random.uniform(min_x, max_x, (dataSets, features))
# Join the x0 column
x0Column = np.ones(( dataSets, 1))
xMatrix = np.concatenate( (x0Column, xMatrix), axis=1)
debug( {"x matrix" : xMatrix}, "X values created" )


###  Aufgabe 2
def vector_linear_hypothesis(thetaVector):
    def costFunc(xFeatureVector):
        debug( {"Theta" : thetaVector, "Feature vector": xFeatureVector}, "In linear hypothesis" )
        return np.dot(xFeatureVector, thetaVector)
    return costFunc

h = vector_linear_hypothesis(thetas)
debug( {"Exakte Thetas" : original_thetas} )

###  Aufgabe 3
# Generate y values using the linear hypothesis function
y = np.array( [ h(featureVector) for featureVector in xMatrix ] )
debug( {"Y before noise" : y })
# Add some noise to it
y += np.random.randn(dataSets) * y_noise_intensity
debug( {"Y noisy" : y })

# Plot as a scatter plot
# ax.scatter(xMatrix[:, 1], xMatrix[:, 2], y, "b", s=60, marker="*")
scatterPlot = plotty.ScatterPlotty3D(xMatrix[:,1], xMatrix[:,2], y, "Randomly generated X features with calculated Y value")
scatterPlot.setLabel("Feature 1", "Feature 2", "Generated  Y values")
scatterPlot.show()
# scatterPlot(xMatrix, y)

def cost_function(xMatrix, y):
    m = len(xMatrix)
    def squared_error_cost(thetaVector):
        loss = vector_linear_hypothesis(thetaVector)(xMatrix) - y
        debug( {"x matrix": xMatrix, "y values" : y, "loss" : loss}, "In cost function" )
        return 1. / ( 2. * m ) * ( loss ** 2 ).sum()
    return squared_error_cost

j = cost_function(xMatrix, y)

# Merke: theta.transpose() * xi == h_theta(x)
def compute_new_theta(xMatrix, y, thetas, alpha):
    m = len(xMatrix)
    leSum = xMatrix.transpose().dot( vector_linear_hypothesis(thetas)(xMatrix) - y )
    thetas_neu = thetas - alpha * (1. / m) * leSum
    debug( {"New theta" : thetas_neu}, "In compute new theta" )
    return thetas_neu

# Do 10.000 iterations
for i in range(iterations):
    thetas = compute_new_theta(xMatrix, y, thetas, alpha)
    costs[i] = cost_function(xMatrix, y)(thetas)

print("Done after {} iterations.".format(iterations))
print("{0} I assume, that I have found something close to the original theta.{0}\n \
        I guess it must have been around.. \n {1}".format( (30*"-"), np.vectorize("%.2f".__mod__)(thetas)))

print("(Original thetas were {}".format(original_thetas))

