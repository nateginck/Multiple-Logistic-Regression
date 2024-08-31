# Nate Ginck
# HW 2: Intro to ML

import numpy as np
import pandas as pd

# cost function:
# X is vector of variables
# theta is vector of estimates
# y is vector of response
def computeCost(X, y, theta):
    # compute X*theta
    estimate = X @ theta

    # compute error
    error = estimate-y
    error_squared = np.square(error)

    # find sum of all the error
    SSE = np.sum(error_squared)

    # find average
    MSE = (1/(2*len(error_squared)))*SSE

    # return cost
    return MSE


# enter toy dataset
toy = np.array([[1,1,1,8],[1,2,2,6],[1,3,3,4],[1,4,4,2]])

# theta values
theta1 = [0, 1, 0.5]
theta2 = [10, -1, -1]
theta3 = [3.5, 0, 0]

# compute cost if main for the three estimates
if '__main__' == __name__:
    J1 = computeCost(toy[:,:3], toy[:,3], theta1)
    print("Cost of J1: ", J1)
    J2 = computeCost(toy[:,:3], toy[:,3], theta2)
    print("Cost of J2: ", J2)
    J3 = computeCost(toy[:,:3], toy[:,3], theta3)
    print("Cost of J3: ", J3)