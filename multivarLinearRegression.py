#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

separator = 30*"-"+"\n"

### Aufgabe 1
m = dataSets = 100
n = features = 4
min_x = -10
max_x = 10
y_noise_intensity = 2

# Create x values
x = np.random.uniform(min_x, max_x, (dataSets, features))
# Join the x0 column
x0Column = np.ones(( m, 1))
x = np.concatenate( (x0Column, x), axis=1)

###  Aufgabe 2
def vector_linear_hypothesis(thetaVector):
    def costFunc(featureVector):
        print(str(featureVector.shape))
        print(str(thetaVector.shape)) 
        print("Got featureVector of shape %s and have theta of shape %s " % (str(featureVector.shape), str(thetaVector.shape)))
        result = np.dot(featureVector, thetaVector)
        return result
    return costFunc

original_thtas = thetas = np.array( [1.1, 2.0, -.9, 0.1, 1.0] )
h = vector_linear_hypothesis(thetas)
print("Thetas shape: %s , x shape: %s " % (thetas.shape, x.shape))

###  Aufgabe 3
# Generate y values using the linear hypothesis function
y = []
for dataSet in x:
    print("Calculating %s" % str(dataSet))
    y.append(h(dataSet))
# And store in a numpy array
y = np.array(y)

print("Y before noise: " + separator)
print(y)

# Add some noise to it
y_noise = np.random.randn(dataSets) * y_noise_intensity
y += y_noise

print("Y after noise: " + separator)
print(y)

# Plot as a scatter plot
print("Plotting:" + separator)
print("x1: " + str(x[:, 1]))
print("x2: " + str(x[:, 2]))
print("y: " + str(y))

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 1], x[:, 2], y, "b", s=60)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 1')
ax.set_zlabel('Target')
# plt.show()

def cost_function(x, y):
    print("Cost function got x vector with shape %s and y with shape %s" % ( str(x.shape), str(y.shape) ))
    m = len(x)
    def squared_error_cost(thetaVector):
        loss = vector_linear_hypothesis(thetaVector)(x) - y
        print("Loss (" + str(loss.shape) + ": " + str(loss))
        return 1. / ( 2. * m ) * ( loss ** 2 ).sum()
    return squared_error_cost

j = cost_function(x, y)
print( j(thetas) )

# Merke: theta.transpose() * xi == h_theta(x)
def compute_new_theta(x, y, thetas, alpha):
    m = len(x)
    print("Old thetas shape: %s , x shape: %s" % (str(thetas.shape), str(x.shape)))
    leSum = x.transpose().dot( vector_linear_hypothesis(thetas)(x) - y )
    thetas_neu = thetas - alpha * (1. / m) * leSum
    print("New thetas (%s): %s" % (str(thetas.shape), str(thetas)))
    return thetas_neu
    

iterations = 1000
alpha = 0.01
costs = {}
# Do 10.000 iterations
for i in range(iterations):
    thetas = compute_new_theta(x, y, thetas, alpha)
    costs[i] = cost_function(x, y)(thetas)

print(separator)
print("I assume:")
print("thetas: " + str(thetas))
print("(Correct was " + str(original_thtas) )
print(separator)

