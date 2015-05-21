# .vim/colors/ | grep green
/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt

# Base values for the line
a = 8
b = 5
noise_offset = 2

# Create x values
xValuesCount = 10
x = np.random.uniform(-10, 10, xValuesCount)

# straight line a*x + b
y = a + b * x
print("Actual function: %s * x + %s" % (a, b) )
plt.figure(1)
plt.plot(x, y, "r")
# plt.title("Initial function")

# Create and apply noise values
y_noise = np.random.randn(xValuesCount) * noise_offset
y += y_noise

print(30*"-")
print("X values at start")
print(x)
print("Y values at start")
print(y)
print(30*"-")

plt.plot(x, y, "b*")
plt.title("Randomly distributed points around this function")

# Define our hypothesis theta0 + theta1*x
# We want to learn to two theta values
def linear_hypothesis(theta0, theta1):
    def costFunc(x):
        print("Got x: ")
        print(x)
        return theta0 + theta1*x
    return costFunc


# def cost_function(x, y):
#     m = len(x)
#     return lambda theta0, theta1: 1. / (2. * m) * ((linear_hypothesis(theta0, theta1)(x) - y) ** 2).sum()

def cost_function(x, y):
    m = len(x)
    def squared_error_cost(theta0, theta1):
        h = linear_hypothesis(theta0, theta1)
        return 1. / ( 2. * m ) * ( (h(x) - y) ** 2 ).sum()
    return squared_error_cost

# Plot the cost function as a contour plot
plotRange = 4
# Plot everything around the actually correct thetas
contour_theta0 = np.arange(a - plotRange, a + plotRange, plotRange * 0.05)
contour_theta1 = np.arange(b - plotRange, b + plotRange, plotRange * 0.05)

C = np.zeros([len(contour_theta0),len(contour_theta1)])
c =  cost_function(x, y)

# Populate the C array with the cost function values (optimal are exactly centered)
for i, t_0 in enumerate(contour_theta0):
    for j, t_1 in enumerate(contour_theta1):
        C[j][i] = c(t_0, t_1)

T0, T1 = np.meshgrid(contour_theta0, contour_theta1)
# plt.subplot(232)
plt.figure(2)
plt.contour(T0, T1, C)
plt.xlabel('$\Theta_0$')
plt.ylabel('$\Theta_1$')
plt.title('Kostenfunktion')

# Alpha is the learning rate
def compute_new_theta(x, y, oldTheta0, oldTheta1, alpha):
    m = len(x)
    newTheta0 = oldTheta0 - alpha * ( ( 1. / m ) * ( oldTheta0  + oldTheta1 * x - y ).sum() )
    newTheta1 = oldTheta1 - alpha * ( ( 1. / m ) * ( (oldTheta0 + oldTheta1 * x - y) * x ).sum() )
    return newTheta0, newTheta1

theta_0 = 1
theta_1 = 2
alpha = 0.001
costs = {}
iterations = 10000

# Do 10.000 iterations
for i in range(iterations):
    theta_0, theta_1 = compute_new_theta(x, y, theta_0, theta_1, alpha)
    costs[i] = cost_function(x, y)(theta_0, theta_1)

# Done finding thetas!
print("After %s iterations, the approximate function is:" % iterations)
print("%s * x + %s" % (theta_0, theta_1) )

# Show the decrease of the cost function with ongoing iteration (e.g. iterations count X axis, cost Y axis)
plt.figure(3)
plt.plot(costs.keys(), costs.values(), 'r-')
plt.xlabel('Iterations')
plt.ylabel('Square error cost')
plt.title('Decrease of error cost throughout the update iterations')

# x = np.array([-10., 10.])
# learnedFunction = lambda x: theta_0 + theta_1 * x
learnedFunction = theta_0 + theta_1 * x
plt.figure(4)
plt.plot(x, y, "b*")
plt.plot(x, learnedFunction, "r")
plt.title("Learned function over " + str(iterations ) + " iterations")
plt.show()

# plt.plot(x, y, 'b*')        # Blaue Sterne als unsere gausswerte
# x_ = np.array([-10., 10.])
# h_ = theta_0 + theta_1 * x_
# plt.plot(x_, h_, 'r-')      # Die optimalste Gerade nach 20000 Iterationen
# plt.show()


# def compute_new_theta(x, y, theta0, theta1, alpha):
#     m = len(x)
#     # update rules
#     t0 = theta0  - alpha * 1./float(m) * (theta0 + theta1 * x - y).sum()
#     t1 = theta1  - alpha * 1./float(m) * ((theta0 + theta1 * x - y) * x ).sum()
#     return t0, t1
#
# theta_0, theta_1 = compute_new_theta(x, y, theta_0, theta_1, alpha)
#
#
#
#
#



