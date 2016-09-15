import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import make_scorer,accuracy_score
from sklearn.grid_search import GridSearchCV
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

parameters = {'n_neighbors': [3,5,7,9],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],'leaf_size':[20,30,40]}
cv_sets = cross_validation.ShuffleSplit(NQuantX.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
reg =  KNeighborsRegressor()
grid_obj = GridSearchCV(reg,param_grid=parameters,cv=cv_sets)
grid_obj=grid_obj.fit(NQuantX, y)

print grid_obj.best_estimator_.get_params()

results = reg.predict(NQuantTest)
output['SalePrice'] = results
output.to_csv('Vayne2.csv', index=False)
end=time.clock()

print "Cost {:.4f}s.".format(end-start)
