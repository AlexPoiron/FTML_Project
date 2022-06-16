import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

#
# Load data
#
data = scale(np.load("./inputs.npy"))
labels = np.load("./labels.npy")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=524)
#
# Fit a pipeline using Training dataset and related labels
# Use Lasso algorithm for training the model
#
alphas = 10 ** np.linspace(10, -2, 100) * 0.5

lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
lassocv = LassoCV(alphas=None, cv=cv, max_iter=100000)
lassocv.fit(X_train, np.ravel(y_train))

pipeline = make_pipeline(StandardScaler(), Lasso(alpha=lassocv.alpha_))
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
