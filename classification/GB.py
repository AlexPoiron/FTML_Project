import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import metrics

data = np.load("./inputs.npy")
labels = np.load("./labels.npy")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.22, random_state=1)

clf = GradientBoostingClassifier(n_estimators=1100, loss="log_loss", criterion="friedman_mse", learning_rate=0.015, max_depth=5, random_state=0)

clf = clf.fit(scale(X_train), np.ravel(y_train))

y_pred = clf.predict(scale(X_test))

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
