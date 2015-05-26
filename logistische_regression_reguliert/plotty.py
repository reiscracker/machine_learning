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
    axe.scatter( x1, x2, y, c="r", s=50 )
    axe.set_xlabel("Feature 1")
    axe.set_ylabel("Feature 2")
    axe.set_zlabel("Generated  Y values")
    plt.title(title)

def plot_costs(costs):
    plt.figure()
    plt.plot(costs.keys(), costs.values(), 'r-')
    plt.xlabel('Iterations')
    plt.ylabel('Square error cost')
    plt.title('Decrease of error cost throughout the update iterations')

def plot_result_plain(x1, x2, y, result_theta, h):
    fig = plt.figure(figsize=(11, 8), dpi=160)
    title = "Randomly generated X features with calculated Y value"
    axe = fig.add_subplot(111, projection='3d', label=title)

    axe.set_xlabel("Feature 1")
    axe.set_ylabel("Feature 2")
    axe.set_zlabel("Generated  Y values")
    plt.title("Learned plain from x1,x2,y data sets")

    # Create meshgrid to plot a surface
    plainX, plainY = np.meshgrid(np.linspace(-10,10), np.linspace(-10,10))

    # Von Lukas Hoedel kopiert
    xMatrix  = np.array( [ np.array([1., xx, yy]) for xx, yy in zip(np.ravel(plainX), np.ravel(plainY)) ] )
    plainZ = h(xMatrix)
    plainZ = plainZ.reshape(plainX.shape)
    axe.plot_surface(plainX, plainY, plainZ, cmap=cm.gray, linewidth=0 )

    # Eigene Loesung
    # Ebenengleichung zur Darstellung der 2-Feature Ebene
#     plainFunc = lambda x, y: result_theta[0] + result_theta[1]*x + result_theta[2]*y
#     axe.plot_surface(plainX, plainY, plainFunc(plainX, plainY), cmap=cm.gray, linewidth=0 )
    # Plot the scatter on top
    axe.scatter( x1, x2, y, c="r", s=60 )

def show_plots():
    plt.show()

