##
import numpy as np
import pandas as pd
import seaborn as sns

in_file = 'train.csv'
full_data = pd.read_csv(in_file)
full_data.head()

##
full_data.describe()

##
full_data['Age']=full_data['Age'].fillna(full_data['Age'].median())
full_data['Name']=full_data['Name'].apply(lambda x:len(x))
full_data['Embarked']=full_data['Embarked'].fillna('S')

full_data['Family']=full_data['SibSp']+full_data['Parch']

full_data.loc[full_data['Sex']=='male','Sex']=0
full_data.loc[full_data['Sex']=='female','Sex']=1
full_data.loc[full_data['Embarked']=='S','Embarked']=0
full_data.loc[full_data['Embarked']=='C','Embarked']=1
full_data.loc[full_data['Embarked']=='Q','Embarked']=2

new_data=full_data.drop(['PassengerId','SibSp','Parch','Cabin','Ticket'], axis = 1)
new_data.head()

import matplotlib.pyplot as plt
%pylab inline
sns.swarmplot(x='Age',y='Sex',hue='Survived',data=full_data)

import matplotlib.pyplot as plt
%pylab inline
sns.barplot(x='Embarked',y='Survived',data=new_data)

import matplotlib.pyplot as plt
%pylab inline
sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=new_data)

import matplotlib.pyplot as plt
%pylab inline
sns.swarmplot(x='Family',y='Age',hue='Survived',data=new_data)


y_all=new_data['Survived']
X_all=new_data.drop('Survived', axis = 1)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=20)

from sklearn.metric import f1_score
def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print "Trained model in {:.4f} seconds".format(end - start)

def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    train_classifier(clf, X_train, y_train)
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))

from sklearn import svm
clf1 = svm.SVC()

from sklearn.neighbors import KNeighborsClassifier
clf2=KNeighborsClassifier()

from sklearn.ensemble import RandomForestClassifier
clf3=RandomForestClassifier()

X_train_1=X_train[:297]
X_train_2=X_train[:594]
X_train_3=X_train[:891]

y_train_1=y_train[:297]
y_train_2=y_train[:594]
y_train_3=y_train[:891]

print "SVM"
train_predict(clf1, X_train_1, y_train_1, X_test, y_test)
train_predict(clf1, X_train_2, y_train_2, X_test, y_test)
train_predict(clf1, X_train_3, y_train_3, X_test, y_test)
print "KNN"
train_predict(clf2, X_train_1, y_train_1, X_test, y_test)
train_predict(clf2, X_train_2, y_train_2, X_test, y_test)
train_predict(clf2, X_train_3, y_train_3, X_test, y_test)
print "RandomForest"
train_predict(clf3, X_train_1, y_train_1, X_test, y_test)
train_predict(clf3, X_train_2, y_train_2, X_test, y_test)
train_predict(clf3, X_train_3, y_train_3, X_test, y_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer,f1_score
from sklearn.grid_search import GridSearchCV
import time
start=time.clock()
parameters = {'n_estimators': [10,20,40,80,120,150,180],'criterion':['gini','entropy']
    ,'max_features':['log2','sqrt',None],'max_depth':[5,6,7,8,9,10],'min_samples_split':[1,2,3]
        ,'warm_start':[False,True]}

clf = RandomForestClassifier()

f1_scorer = make_scorer(f1_score)

grid_obj = GridSearchCV(clf,param_grid=parameters,scoring=f1_scorer)

grid_obj=grid_obj.fit(X_train, y_train)

clf = grid_obj.best_estimator_

end=time.clock()

print grid_obj.best_estimator_.get_params()

print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))
print "Optimize model in {:.4f} seconds".format(end - start)
