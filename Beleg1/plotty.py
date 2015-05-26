#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def scatter(x1, x2, y):
    if ( x1.shape[0] != x2.shape[0] != y.shape[0]):
        raise("Invalid dimensions of operand!")

    fig = plt.figure(figsize=(11, 8), dpi=160)
    title = "Randomly generated X features with calculated Y value"
    axe = fig.add_subplot(111, projection='3d', label=title)
    axe.scatter( x1, x2, y, c="r", s=60 )
    axe.set_xlabel("Feature 1")
    axe.set_ylabel("Feature 2")
    axe.set_zlabel("Generated  Y values")
    plt.title(title)
#     plt.show()

def plot_costs(costs):
    plt.figure()
    plt.plot(costs.keys(), costs.values(), 'r-')
    plt.xlabel('Iterations')
    plt.ylabel('Square error cost')
    plt.title('Decrease of error cost throughout the update iterations')
#     plt.show()

def plot_result_plain(x1, x2, y, result_theta):
    fig = plt.figure(figsize=(11, 8), dpi=160)
    title = "Randomly generated X features with calculated Y value"
    axe1 = fig.add_subplot(111, projection='3d', label=title)

    axe1.scatter( x1, x2, y, c="r", s=60 )

    plainFunc = lambda x, y: result_theta[0] + result_theta[1]*x + result_theta[2]*y
    plainX, plainY = np.meshgrid(range(-10, 10), range(-10, 10))
    print "XYZ: {} {} {}".format(plainX, plainY, plainFunc(plainX, plainY))
    print "Shapes: {} {} {}".format(plainX.shape, plainY.shape, plainFunc(plainX, plainY).shape)
    axe1.plot_surface(plainX, plainY, plainFunc(plainX, plainY), cmap=cm.gray, linewidth=0 )
    plt.show()

# def scatterPlot2D( x, y ):
#     if ( xMatrix.shape[0] != y.shape[0] ):
#         raise("Invalid dimensions of operand!")
# 
#     axis =     axis.scatter(xMatrix[:, 1], xMatrix[:, 2], y, "b", s=60, marker="*")
#     axis.set_xlabel('Feature 1')
#     axis.set_ylabel('Feature 2')
#     axis.set_zlabel('Target')
#     plt.show()
# 
# def myPlot( xMatrix, y ):
#     if ( xMatrix.shape[0] != y.shape[0] ):
#         raise("Invalid dimensions of operand!")
# 
#     fig = plt.figure(figsize=(11, 8), dpi=160)
#     ax = fig.add_subplot(111, projection='3d', label="Zufaellige Datenmatrix mit kuenstlich erzeugenten Y-Werten")
#     ax.scatter(xMatrix[:, 1], xMatrix[:, 2], y, "b", s=60, marker="*")
#     ax.set_xlabel('Feature 1')
#     ax.set_ylabel('Feature 2')
#     ax.set_zlabel('Target')
#     plt.show()

# def scatterPlot3D( x, y, z ):
#     if ( xMatrix.shape[0] != y.shape[0] ):
#         raise("Invalid dimensions of operand!")
# 
#     fig = plt.figure(figsize=(11, 8), dpi=160)
#     ax = fig.add_subplot(111, projection='3d', label="Zufaellige Datenmatrix mit kuenstlich erzeugenten Y-Werten")
#     ax.scatter(xMatrix[:, 1], xMatrix[:, 2], y, "b", s=60, marker="*")
#     ax.set_xlabel('Feature 1')
#     ax.set_ylabel('Feature 2')
#     ax.set_zlabel('Target')
#     plt.show()
