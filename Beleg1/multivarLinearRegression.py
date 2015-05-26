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
    return d if enabled else lambda a,b=None:1

def vector_linear_hypothesis(thetaVector):
    def costFunc(xFeatureVector):
        return np.dot(xFeatureVector, np.transpose(thetaVector))
    return costFunc

## 4. Aufgabe
def cost_function(xMatrix, y):
    m = len(xMatrix)
    def squared_error_cost(thetaVector):
        loss = vector_linear_hypothesis(thetaVector)(xMatrix) - y
        return 1. / ( 2. * m ) * ( loss ** 2 ).sum()
    return squared_error_cost

## 5. Aufgabe
# Merke: theta.transpose() * xi == h_theta(x)
def compute_new_theta(xMatrix, y, thetas, alpha):
    m = len(xMatrix)
    leSum = xMatrix.transpose().dot( vector_linear_hypothesis(thetas)(xMatrix) - y )
    thetas_neu = thetas - alpha * (1. / m) * leSum
    return thetas_neu

def gradient_decent(alpha, theta, nb_iterations, X, y):
    # Do 10.000 iterations
    costs = {}
    for i in range(nb_iterations):
        debug( {"thetas": theta}, "After %s iterations"  % i )
        theta = compute_new_theta(xMatrix, y, theta, alpha)
        costs[i] = cost_function(xMatrix, y)(theta)
    return theta, costs

if __name__ == "__main__":
    isDebug = (len(sys.argv) > 1 and "-d" in sys.argv)
    debug = get_debug_function(isDebug)

    dataSets, features = 100,  2
    min_x, max_x = -10., 10.
    original_thetas = np.array( [1.1, 2.0, -0.9] )
    start_thetas = np.array( [ 1.5, 1.0, 0.0 ] )
    y_noise_intensity = 3.
    iterations = 1000
    alpha = 0.0001

    ### Aufgabe 1
    # Create x values
    xMatrix = np.random.uniform(min_x, max_x, (dataSets, features))
    # Join the x0 column
    x0Column = np.ones(( dataSets, 1 ))
    xMatrix = np.concatenate( (x0Column, xMatrix), axis=1)
    debug( {"X matrix" : xMatrix}, "X values created" )


    ###  Aufgabe 2
    h = vector_linear_hypothesis(original_thetas)
    debug( {"Exakte Thetas" : original_thetas} )

    ###  Aufgabe 3
    # Generate y values using the linear hypothesis function
    # y = np.array( [ h(featureVector) for featureVector in xMatrix ] )
    y = h(xMatrix)
    debug( {"Y before noise" : y })
    # Add some noise to it
    y += np.random.randn(dataSets) * y_noise_intensity
    debug( {"Y noisy" : y })

    # Plot as a scatter plot
    plotty.scatter(xMatrix[:,1], xMatrix[:,2], y)

    ## 5. Aufgabe
    result_thetas, costs =  gradient_decent(alpha, start_thetas, iterations, xMatrix, y)
    plotty.plot_costs(costs)
    ## 6. Aufgabe
    plotty.plot_result_plain(xMatrix[:,1], xMatrix[:,2], y, result_thetas, vector_linear_hypothesis(result_thetas))

    print("Done after {} iterations.".format(iterations))
    print("{0} I assume, that I have found something close to the original theta.{0}\n \
            I guess it must have been around.. \n {1}".format( (30*"-"), result_thetas))

    print("(Original thetas were {} ".format(original_thetas))

    plotty.show_plots()
