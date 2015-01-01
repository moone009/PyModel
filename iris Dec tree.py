# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 22:00:59 2014

@author: christopher
"""

from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


# import some data to play with

iris = pd.read_csv('/home/christopher/iris.csv',sep=',',header=False)

iris.columns = ["A", "B", "C", "D","TYPE"]


# import some data to play with

iristest = pd.read_csv('/home/christopher/iristest.csv',sep=',',header=True)

iristest.columns = ["A", "B", "C", "D","TYPE"]


# fit a CART model to the data

model = DecisionTreeClassifier()

model.fit(iris[["A", "B", "C", "D"]], iris[["TYPE"]])

print(model)

# make predictions

expected = iristest.TYPE

predicted = model.predict(iristest[["A", "B", "C", "D"]])

predicted = pd.DataFrame(predicted)


# reset indexes

predicted = predicted.reset_index()

expected = expected.reset_index()


# summarize the fit of the model

Final = predicted.merge(expected,on='index')

Final.columns = ["index", "Pred", "Actual"]

Final['Correct'] = (Final['Pred'] == Final['Actual'])


Summ = Final.groupby('Correct').count()

print(Summ)

print(pd.crosstab(Final.Pred, Final.Correct).apply(lambda r: r/r.sum(), axis=1))

print(pd.crosstab(Final.Pred, Final.Actual).apply(lambda r: r/r.sum(), axis=1))

print(pd.crosstab(Final.Pred, Final.Actual))


# Accuracy and Recall and Precision

Accuracy = (14+13+16) /(14+0+0+0+13+3+0+3+16)

RecallSet = 14 /(14+0+0)

Recallversicolor = 13 /(0+13+3)

Recallvirginica = 16 /(0+3+16)


precisionsetosa = 14/(14+0+0)

precisionversicolor =13/(0+13+3)

precisionvirginica =16/(0+3+16)



Precision = TP / (TP+FP)

Recall = TP / (TP+FN)


