import numpy as np
import pandas as pd

from numpy.random import uniform
from numpy.random import normal

X = np.random.randint(3, size=100000)

def create_y(X):
    y = []
    for value in X:
        if value < 1:
            y.append(uniform(0,1,size=None))
        else:
            y.append(normal(value,size=None))
    return y 

y = create_y(X)

def predict_bayes(X):
    y = []
    for value in X:
        if value < 1:
            y.append(1/2)
        else:
            y.append(value)
    return y

def predict_tilde(X):
    return X

f_bayes = predict_bayes(X)
f_tilde = predict_tilde(X)

def statistical_approximation(y_true, y_pred):
    vec = []
    for i in range(len(y_true)):
        vec.append((y_true[i] - y_pred[i])**2)
    return np.mean(vec)

stat_val_tilde = statistical_approximation(y, predict_tilde(X))
real_bayes_risk = 1/24+2/3

print("Using the statistical approximation of the tilde risk which is:", stat_val_tilde)
print("We also computed the exact value of the bayes risk which is:", real_bayes_risk)