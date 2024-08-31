import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from Reg_normalEqn import Reg_normalEqn
from computeCost import computeCost
from logReg_multi import logReg_multi


# 1b: Load hw4_data1.mat
data1 = scipy.io.loadmat(r'input\hw4_data1.mat')

# store data as variables
X = np.array(data1['X_data'])
y = np.array(data1['y'])

# add offset feature
X = np.insert(X, 0, 1, axis=1)

# print size of feature matrix
print("The size of X is: ", X.shape)

# combine X and y to shuffle together
Xy = np.hstack((X, y))

# define Lambda array
Lambda = [0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017]

# define test and training MSE arrays
TrainingCost = np.empty([20, 8])
TestCost = np.empty([20, 8])

# 1c: Compute average training and testing error
for i in range(20):
    # shuffle data and split into training and testing
    np.random.shuffle(Xy)

    split = int(Xy.shape[0] * 0.85)

    # split data using this value, split
    X_train = Xy[:split, :Xy.shape[1] - 1]
    y_train = Xy[:split, Xy.shape[1] - 1]

    X_test = Xy[split:, :Xy.shape[1] - 1]
    y_test = Xy[split:, Xy.shape[1] - 1]

    # train 8 regression model models
    for j in range(8):
        # train and calculate cost for training error
        temp_parameters = Reg_normalEqn(X_train, y_train, Lambda[j])
        TrainingCost[i, j] = computeCost(X_train, y_train, temp_parameters)

        # calculate cost for testing set
        TestCost[i, j] = computeCost(X_test, y_test, temp_parameters)


# calculate averages plot results from iterations
TestAverage = np.average(TestCost, axis=0)
TrainingAverage = np.average(TrainingCost, axis=0)

# plot results
plt.plot(Lambda, TestAverage, label="Testing Error", color="blue", linestyle="-", marker="o")
plt.plot(Lambda, TrainingAverage, label="Training Error", color="red", linestyle="-", marker="o")
plt.xlabel("Lambda")
plt.ylabel("Average Error")
plt.legend(loc="upper right")
plt.savefig('output/ps4-1-a.png')
plt.close()

# 2a: load in Matlab data
data2 = scipy.io.loadmat(r'input\hw4_data2.mat')
# store data
X1 = np.array(data2['X1'])
X2 = np.array(data2['X2'])
X3 = np.array(data2['X3'])
X4 = np.array(data2['X4'])
X5 = np.array(data2['X5'])
y1 = np.array(data2['y1'])
y2 = np.array(data2['y2'])
y3 = np.array(data2['y3'])
y4 = np.array(data2['y4'])
y5 = np.array(data2['y5'])

# combine into a single variable
X = [X1,X2,X3,X4,X5]
y = [y1,y2,y3,y4,y5]

# set K
K = list(range(1, 16, 2))
accuracy = np.empty([8, 5])

# iterate through 5-fold cross validation
for i in range(len(K)):
    for j in range(5):
        # set aside one of the folds for testing
        X_test = X[j]
        y_test = y[j]
        # concatenate training data
        X_train = np.concatenate([X[z] for z in range(len(X)) if z != j])
        y_train = np.concatenate([y[z] for z in range(len(y)) if z != j])
        y_train = y_train.reshape(y_train.size,)

        # fit model onto training data
        model = KNeighborsClassifier(n_neighbors=K[i])
        model.fit(X_train, y_train)

        # predict model
        y_pred = model.predict(X_test)

        # calculate accuracy
        accuracy[i][j] = accuracy_score(y_test, y_pred)

# calculate accuracy average
accuracy_average = np.average(accuracy, axis=1)

# save as png
plt.plot(K, accuracy_average, color='black', marker='o')
plt.xlabel("K values")
plt.ylabel("Average Accuracy values")
plt.savefig("output/ps4-2-a.png")
plt.close()

# 3a: read in matlab file
data3 = scipy.io.loadmat(r'input/hw4_data3.mat')

# store data
X_train = np.array(data3['X_train'])
X_test = np.array(data3['X_test'])
y_train = np.array(data3['y_train'])
y_test = np.array(data3['y_test'])

# 3b:
# run function to find predictions
y_predict_train = logReg_multi(X_train, y_train, X_train)
y_predict_test = logReg_multi(X_train, y_train, X_test)

# compute accuracy of training and testing accuracy
accuracy_train = accuracy_score(y_train, y_predict_train)
accuracy_test = accuracy_score(y_test, y_predict_test)

# write as table
print("Train,", "Test")
print(accuracy_train, accuracy_test)