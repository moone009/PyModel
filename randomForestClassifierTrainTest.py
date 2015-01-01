# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 15:15:03 2014

@author: christopher
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from pandasql import sqldf
from pandasql import load_meat, load_births
pysqldf = lambda q: sqldf(q, globals()) 

from sklearn.cross_validation import train_test_split

def randomForestClassifierTrainTest(train,test,Features,TargetVariable):
    

    Features = Features
    TargetVariable = TargetVariable
    test = test
    train = train
    features = train.columns[0:Features]

    clf = RandomForestClassifier(n_jobs=5000)
    y, _ = pd.factorize(train[TargetVariable])
    ClassNames = pd.factorize(train[TargetVariable])
    clf.fit(train[features], y)
    #preds = iris[TargetVariable][clf.predict(test[features])]
    #preds = clf.predict(test[features])
    preds = ClassNames[1][clf.predict(test[features])]
    rows_list = []

    importances = clf.feature_importances_
    important_names = features[importances > np.mean(importances)]
    print('Import Cols:',important_names)

    for row in preds:
        rows_list.append(row)
    
    df = pd.DataFrame(rows_list)
    df.columns =['Predicted Value']
    df = df.reset_index()
    print('df rows:',len(df))

    p = test
    p = p.reset_index(drop=True)
    p = p.reset_index()
    print('p rows:',len(p))
    merged = df.merge(p,on='index')
    print('merged rows:',len(merged))

    return(merged)
    
    
# Load Data
iris = pd.read_csv('/home/christopher/iris.csv')
q = "Select *  from iris order by random()"
iris = pysqldf(q) 

# Split into test and train
msk = np.random.rand(len(iris)) < 0.6

train = iris[msk]
test = iris[~msk]

# Run
Test = randomForestClassifierTrainTest(train,test,3,'Species')
pd.crosstab(Test['Species'],Test['Predicted Value'])
    
    
    
    
    
    
    
    
    