# Nathaniel Ginck
import numpy as np

# Using Ridge Regularization for a closed solution
def Reg_normalEqn(X_train, y_train, Lambda):
    # make an Identity Matrix I of size X'X
    X_temp = X_train.transpose() @ X_train
    I = np.eye(X_temp.shape[0])

    # set first 1 to 0 such that the intercept is not regularized
    I[0,0] = 0

    # return normal equation for adjusted data
    return np.linalg.pinv((X_temp + Lambda * I)) @ X_train.transpose() @ y_train
