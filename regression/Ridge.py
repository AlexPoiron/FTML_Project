import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

#
# Load data
#
data = scale(np.load("./inputs.npy"))
labels = np.load("./labels.npy")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=524)
#
# Fit a pipeline using Training dataset and related labels
# Use Ridge algorithm for training the model
# Genretaing alphas from a very large interval
#
alphas = 10 ** np.linspace(10, -2, 100) * 0.5

ridge = Ridge()
coefs = []

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_[0])

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()

cv = RepeatedKFold(n_splits=8, n_repeats=5, random_state=81)
ridgecv = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')

ridgecv.fit(X_train, y_train)
print(ridgecv.alpha_)
pipeline = make_pipeline(StandardScaler(), Ridge(alpha=ridgecv.alpha_))
pipeline.fit(X_train, y_train)
#
# Calculate the predicted value for training and test dataset
#
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
#
# Mean Squared Error
#
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
#
# R-Squared
#
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
