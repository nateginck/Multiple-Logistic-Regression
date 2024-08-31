import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

def logReg_multi(X_train, y_train, X_test):
    # determine how many classifications exist in y
    classes = np.unique(y_train)
    # create y_c vector
    y_c = np.zeros((y_train.shape[0], len(classes)))
    # encode binary response
    for i in range(len(y_train)):
        val = y_train[i]
        y_c[i, val - 1] = 1

    # create models for each class and train
    models = {f"model_{i}": LogisticRegression(random_state=0).fit(X_train, y_c[:, i]) for i in range(len(classes))}
    predictions = {}

    # calculate probability for each model with training data
    for model_name, model in models.items():
        proba = model.predict_proba(X_test)
        predictions[model_name] = proba[:, 1]

    # combine dictionary to array with all positive probabilities
    probabilities = np.column_stack(list(predictions.values()))

    # based on highest probability, map to 1, 2, or 3
    y_predict = np.zeros(len(probabilities))
    for i in range(len(probabilities)):
        y_predict[i] = np.argmax(probabilities[i]) + 1

    # return 1D vector
    return y_predict
