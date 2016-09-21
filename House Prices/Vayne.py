import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor

start=time.clock()
train = pd.read_csv("train.csv")
output = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")

output.index = range(len(output))

y = train['SalePrice']
X = train.drop(['SalePrice'], axis=1)

quantX = X._get_numeric_data()
quantTest = test._get_numeric_data()

quantX = quantX.fillna(0)
quantTest = quantTest.fillna(0)

scaledQuantX = preprocessing.scale(quantX)
NQuantX = pd.DataFrame(scaledQuantX)

scaledQuantTest = preprocessing.scale(quantTest)
NQuantTest = pd.DataFrame(scaledQuantTest)

knn =  KNeighborsRegressor(n_neighbors = 3,algorithm='brute',p=2)
knn.fit(NQuantX, y)

results = knn.predict(NQuantTest)
output['SalePrice'] = results
output.to_csv('Vayne7.csv', index=False)
end=time.clock()

print "Cost {:.4f}s.".format(end-start)
