import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import metrics

data = np.load("./inputs.npy")
labels = np.load("./labels.npy")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.35, random_state=4042)

clf = SGDClassifier(max_iter=1000, loss="hinge", tol=0.001, alpha=0.001, random_state=967)

clf = clf.fit(scale(X_train), np.ravel(y_train))

y_pred = clf.predict(scale(X_test))

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
