#!/usr/bin/python

import numpy as np
import plotty
import sys

def get_debug_function(enabled=False):
    def d(debugValues, caller=None):
        debugHeader = "Debug message from {0} {1}".format
        debugElements = "Element {n} of type {t} (Shape: {v.shape}):\n{v}".format
        print( debugHeader(caller if caller else "None", 30*"-") )
        print( "\n".join( [ debugElements(n=name, v=value, t=type(value)) for name, value in debugValues.iteritems() ] ) )
    if not enabled:
        # Return dummy function so no debug is printed
        return lambda debugValues, caller=None: 1+1
    else:
        return d
isDebug = (len(sys.argv) > 1 and "-d" in sys.argv)
debug = get_debug_function(isDebug)


dataSets, features = 100,  2
min_x, max_x = -10., 10.
original_thetas = np.array( [1, 2, 3] )
thetas = np.array( [ -0.3, 3, 3.17 ] )
y_noise_intensity = 2
iterations = 10000
alpha = 0.001
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

h = vector_linear_hypothesis(original_thetas)
debug( {"Exakte Thetas" : original_thetas} )

###  Aufgabe 3
# Generate y values using the linear hypothesis function
y = np.array( [ h(featureVector) for featureVector in xMatrix ] )
debug( {"Y before noise" : y })
# Add some noise to it
# y += np.random.randn(dataSets) * y_noise_intensity
debug( {"Y noisy" : y })

# Plot as a scatter plot
plotty.scatter(xMatrix[:,1], xMatrix[:,2], y)

def cost_function(xMatrix, y):
    m = len(xMatrix)
    def squared_error_cost(thetaVector):
        loss = vector_linear_hypothesis(thetaVector)(xMatrix) - y
        debug( {"x matrix": xMatrix, "y values" : y, "loss" : loss}, "In cost function" )
        return 1. / ( 2. * m ) * ( loss ** 2 ).sum()
    return squared_error_cost

# j = cost_function(xMatrix, y)

# Merke: theta.transpose() * xi == h_theta(x)
def compute_new_theta(xMatrix, y, thetas, alpha):
    m = len(xMatrix)
    leSum = xMatrix.transpose().dot( vector_linear_hypothesis(thetas)(xMatrix) - y )
    thetas_neu = thetas - alpha * (1. / m) * leSum
    debug( {"New theta" : thetas_neu}, "In compute new theta" )
    return thetas_neu

def gradient_decent(alpha, theta, nb_iterations, X, y):
    # Do 10.000 iterations
    for i in range(iterations):
        theta = compute_new_theta(xMatrix, y, theta, alpha)
        costs[i] = cost_function(xMatrix, y)(theta)
    return (theta, costs)

result_thetas, result_costs = gradient_decent(alpha, thetas, iterations, xMatrix, y)
plotty.plot_costs(costs)
# plotty.plot_result_plain(xMatrix[:,1], xMatrix[:,2], y, vector_linear_hypothesis(result_thetas))
plotty.plot_result_plain(xMatrix[:,1], xMatrix[:,2], y, result_thetas)

print("Done after {} iterations.".format(iterations))
print("{0} I assume, that I have found something close to the original theta.{0}\n \
        I guess it must have been around.. \n {1}".format( (30*"-"), result_thetas))

print("(Original thetas were {}".format(original_thetas))

