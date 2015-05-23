#!/usr/bin/python

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plotty:

    def __init__(self, plotTitle, subplots=1):
        self.fig = plt.figure(figsize=(11, 8), dpi=160)
        self.title = plotTitle
        self.axe = self.fig.add_subplot(111, projection='3d', label=plotTitle)

    def show(self):
        plt.title(self.title)
        plt.show()

class ScatterPlotty3D(Plotty):

    def __init__( self, x, y, z, plotLabel, color="b", pointSize=60):
        Plotty.__init__(self, plotLabel)
        if ( x.shape[0] != y.shape[0] != z.shape[0]):
            raise("Invalid dimensions of operand!")
        self.axe.scatter( x, y, z, c=color, s=pointSize, marker="*" )

    def setLabel(self, xLabel, yLabel, zLabel):
        self.axe.set_xlabel(xLabel)
        self.axe.set_ylabel(yLabel)
        self.axe.set_zlabel(zLabel)

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
