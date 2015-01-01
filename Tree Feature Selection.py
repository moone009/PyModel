# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 21:23:27 2014

@author: christopher
"""
http://guidetodatamining.com/chapter-8/
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(iris.data, iris.target)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(4):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(4), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(4), indices)
plt.xlim([-1, 4])
plt.show()