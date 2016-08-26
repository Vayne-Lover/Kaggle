import numpy as np
import pandas as pd
import seaborn as sns

in_file = 'train.csv'
full_data = pd.read_csv(in_file)
full_data.head()

full_data.describe()

full_data.isnull().any()

full_data['Age']=full_data['Age'].fillna(full_data['Age'].median())
full_data['Name']=full_data['Name'].apply(lambda x:len(x))
full_data['Embarked']=full_data['Embarked'].fillna('S')

full_data['Family']=full_data['SibSp']+full_data['Parch']

full_data.loc[full_data['Sex']=='male','Sex']=0
full_data.loc[full_data['Sex']=='female','Sex']=1
full_data.loc[full_data['Embarked']=='S','Embarked']=0
full_data.loc[full_data['Embarked']=='C','Embarked']=1
full_data.loc[full_data['Embarked']=='Q','Embarked']=2

new_data=full_data.drop(['PassengerId','Name','SibSp','Parch','Cabin','Ticket'], axis = 1)
new_data.head()

new_data.describe()

from IPython.display import display
import scipy
for feature in new_data.keys():
    Q1 = np.percentile(new_data[feature],25)
    Q3 = np.percentile(new_data[feature],75)
    step = 2.0*(Q3-Q1)
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(new_data[~((new_data[feature] >= Q1 - step) & (new_data[feature] <= Q3 + step))])

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

import time
from sklearn.metrics import f1_score
def train_classifier(clf, X_train, y_train):
    start = time.clock()
    clf.fit(X_train, y_train)
    end = time.clock()
    print "Trained model in {:.4f} seconds".format(end - start)

def predict_labels(clf, features, target):
    y_pred = clf.predict(features)
    return f1_score(target.values, y_pred)


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

X_train_1=X_train[:230]
X_train_2=X_train[:460]
X_train_3=X_train

y_train_1=y_train[:230]
y_train_2=y_train[:460]
y_train_3=y_train

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

import numpy as np
import pandas as pd
import seaborn as sns
%pylab inline

plot_data=pd.DataFrame({'Name':['SVM','KNN','RDM','Tuned RDM'],
                       'Train Score':[0.880,0.736,0.965,0.882],
                       'Test Score':[0.404,0.574,0.730,0.760]})
sns.pointplot(x='Name',y='Train Score',data=plot_data,markers='o',color='r')
sns.pointplot(x='Name',y='Test Score',data=plot_data,markers='D',color='g')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
                                                            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
                     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
                     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                              label="Training score")
                     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                              label="Cross-validation score")
                     
                     plt.legend(loc="best")
                     return plt

title = "Learning Curves (Random Forest)"

plot_learning_curve(clf, title, X_train, y_train, (0.7, 1.01), cv=cv_sets, n_jobs=2)

plt.show()


