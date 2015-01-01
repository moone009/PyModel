# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 21:13:00 2014

@author: christopher
"""

from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"  % (iris.data.shape[0],(iris.target != y_pred).sum()))

http://scikit-learn.org/stable/modules/neighbors.html
http://scikit-learn.org/stable/modules/tree.html
http://scikit-learn.org/stable/modules/svm.html
http://scikit-learn.org/stable/supervised_learning.html




