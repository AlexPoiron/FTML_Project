import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt

data = np.load("./inputs.npy")
labels = np.load("./labels.npy")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.22, random_state=1)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)

clf = clf.fit(scale(X_train), y_train)

y_pred = clf.predict(scale(X_test))
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,filled=True)
plt.show()

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
