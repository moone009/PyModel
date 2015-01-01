# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 21:33:14 2014

@author: christopher
"""






from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_hastie_10_2
from sklearn.cross_validation import train_test_split

CoverType = pd.read_csv('/home/christopher/Desktop/train.csv')
CoverType  = CoverType.drop('Id',1)    

#CoverType= CoverType[['Elevation', 'Vertical_Distance_To_Hydrology', 'Hillshade_3pm', 'Horizontal_Distance_To_Hydrology', 'Soil_Type3', 'Wilderness_Area4', 'Horizontal_Distance_To_Roadways', 'Aspect', 'Hillshade_9am', 'Hillshade_Noon', 'Horizontal_Distance_To_Fire_Points', 'Soil_Type23', 'Soil_Type7', 'Wilderness_Area3','Cover_Type']]
Numeroffeatures = len(CoverType.columns)-2


Test = randomForestClassifier(CoverType,13,'Cover_Type')
pd.crosstab(Test['Cover_Type'],Test['Predicted Value'])

CoverTypeTest = pd.read_csv('/home/christopher/Desktop/test.csv')
CoverTypeTestID  = CoverTypeTest.Id 
CoverTypeTest  = CoverTypeTest.drop('Id',1)    
CoverTypeTestID  = CoverTypeTestID.reset_index()
#CoverTypeTest= CoverTypeTest[['Elevation', 'Vertical_Distance_To_Hydrology', 'Hillshade_3pm', 'Horizontal_Distance_To_Hydrology', 'Soil_Type3', 'Wilderness_Area4', 'Horizontal_Distance_To_Roadways', 'Aspect', 'Hillshade_9am', 'Hillshade_Noon', 'Horizontal_Distance_To_Fire_Points', 'Soil_Type23', 'Soil_Type7', 'Wilderness_Area3']]









# generate synthetic data from ESLII - Example 10.2
df = CoverType
Features = Numeroffeatures
TargetVariable = 'Cover_Type'
test = CoverTypeTest
train = df
features = df.columns[0:Features]

y, _ = pd.factorize(train[TargetVariable])
ClassNames = pd.factorize(train[TargetVariable])

# fit estimator
est = GradientBoostingClassifier(n_estimators=85, max_depth=1)
est.fit(train[features], y)

# predict class labels
preds = ClassNames[1][est.predict(test[features])]

rows_list = []

for row in preds:
    rows_list.append(row)
    
df = pd.DataFrame(rows_list)
df.columns =['Predicted Value']
df = df.reset_index()   
    
p = test
p = p.reset_index(drop=True)
p = p.reset_index()

mergedSet = df.merge(p,on='index')
merged = mergedSet.merge(CoverTypeTestID,on='index')
merged.to_csv("mergedtest.csv")
merged = ''
est.fit = ''    
    