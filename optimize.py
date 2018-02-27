import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy

#importing classifiers:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#Importing model_selection:
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
#For stacking results from different classifiers:
from sklearn.ensemble import VotingClassifier

train = pd.read_csv(os.path.join('processed', "train.csv"))

X_train = train[['Pclass', 'Sex', 'Age','Fare', 'SibSp', 'Parch', 'Embarked', 'Title']]
y_train = train['Survived']

m = X_train.shape[0]
n = X_train.shape[1]
number_of_estimators = 3 #I'm using 3 different classifiers: logisticRegrassion, AdaBoost, DecisionTree, RandomForests

#Create a numpy array to store the estimators best predictions
predictions_np = np.zeros([m,number_of_estimators])

#Optimize hyperparameters for DecisionTreeClassifier:
DecisionTreeClassifier = DecisionTreeClassifier()

n_folds = 3

tuned_parameters = {'max_depth':scipy.stats.randint(low=6,high=16),'min_samples_leaf':scipy.stats.randint(low=1,high=10) }
# I use Random hyperparameter optimization:
clf1 = RandomizedSearchCV(DecisionTreeClassifier, tuned_parameters, cv=n_folds, refit=True, n_iter=10)
clf1.fit(X_train, y_train)
params_DTC = clf1.best_params_
score = clf1.best_score_
print("____________________________________________")
print("DecisionTreeClassifier best parameters:", params_DTC)
print("DecisionTreeClassifier score:", score)

#predictions_np[:,0] = clf.predict(X_train)

with open('settings.py', 'w') as f:
	f.write('\nDecisionTreeClassifier:')
	f.write(' ' + str(params_DTC))
	f.write(' ' + str(score))

#Optimize hyperparameters for AdaBoostClassifier:
AdaBoostClassifier = AdaBoostClassifier()

n_folds = 3

tuned_parameters = {'learning_rate':scipy.stats.expon(scale=0.1) }
# I use Random hyperparameter optimization:
clf2 = RandomizedSearchCV(AdaBoostClassifier, tuned_parameters, cv=n_folds, refit=True,  n_iter=10)
clf2.fit(X_train, y_train)
params_ABC = clf2.best_params_
score = clf2.best_score_
print("____________________________________________")
print("AdaBoostClassifier best parameters:", params_ABC)
print("AdaBoostClassifier score:", score)

#predictions_np[:,1] = clf.predict(X_train)

with open('settings.py', 'a') as f:
	f.write('\nAdaBoostClassifier:')
	f.write(' ' + str(params_ABC))
	f.write(' ' + str(score))

#Optimize hyperparameters for RandomForestClassifier:
RandomForestClassifier = RandomForestClassifier()

n_folds = 3

tuned_parameters = {'n_estimators': scipy.stats.randint(low=6,high=16), 'max_depth':scipy.stats.randint(low=6,high=16),'min_samples_leaf':scipy.stats.randint(low=1,high=10) }
# I use Random hyperparameter optimization:
clf3 = RandomizedSearchCV(RandomForestClassifier, tuned_parameters, cv=n_folds, refit=True, n_iter=10)
clf3.fit(X_train, y_train)
params_RFC = clf3.best_params_
score = clf3.best_score_
print("____________________________________________")
print("RandomForestClassifier best parameters:", params_RFC)
print("RandomForestClassifier score:", score)

#predictions_np[:,2] = clf.predict(X_train)

with open('settings.py', 'a') as f:
	f.write('\nRandomForestClassifier:')
	f.write(' ' + str(params_RFC))
	f.write(' ' + str(score))


with open('settings.py', 'a') as f:
	f.write('\nBye!!!!')

#Now lets stack the results from the optimized classifiers:

eclf = VotingClassifier(estimators=[('dt', clf1), ('ab', clf2), ('rf', clf3)], voting='hard')


eclf = eclf.fit(X_train,y_train)
print("____________________________________________")
print("VotingClassifier score:")
print(eclf.score(X_train,y_train))
print("____________________________________________")


#OK! and now - let's predict the train dataset survival:
test = pd.read_csv(os.path.join('processed', 'test.csv'))
X_test = test[['Pclass', 'Sex', 'Age','Fare', 'SibSp', 'Parch', 'Embarked', 'Title']]
test['predictions'] = eclf.predict(X_test)
test_kaggle = pd.read_csv(os.path.join('data', 'test.csv'),  usecols = ['PassengerId'])
test_kaggle['Survived'] = test['predictions']
test_kaggle = test_kaggle.set_index('PassengerId')

test_kaggle.to_csv(os.path.join('predictions', "test_predict_kaggle.csv"))

