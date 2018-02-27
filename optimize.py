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

train = pd.read_csv(os.path.join('processed', "train.csv"))

X_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
y_train = train['Survived']

m = X_train.shape[0]
n = X_train.shape[1]
number_of_estimators = 4 #I'm using 4 different classifiers: logisticRegrassion, AdaBoost, DecisionTree, RandomForests

#Create a numpy array to store the estimators best predictions
predictions_np = np.zeros([m,number_of_estimators])

# Optimizing hyperparameters for LogisticRegression
LogisticRegression = LogisticRegression()

n_folds = 3
#The parameter I optimize is C:
tuned_parameters = {'C':scipy.stats.expon(scale=1)}
# I use Random hyperparameter optimization:
clf = RandomizedSearchCV(LogisticRegression, tuned_parameters, cv=n_folds, refit=True)
clf.fit(X_train, y_train)
params = clf.best_params_
score = clf.best_score_
print("____________________________________________")
print("LogisticRegression best parameters:", params)
print("LogisticRegression score:", score)

predictions_np[:,0] = clf.predict(X_train)

with open('settings.py', 'w') as f:
	f.write('LogisticRegression:')
	f.write(' ' + str(params))
	f.write(' ' + str(score))

#Optimize hyperparameters for DecisionTreeClassifier:
DecisionTreeClassifier = DecisionTreeClassifier()

n_folds = 3

tuned_parameters = {'max_depth':scipy.stats.randint(low=6,high=16),'min_samples_leaf':scipy.stats.randint(low=1,high=10) }
# I use Random hyperparameter optimization:
clf = RandomizedSearchCV(DecisionTreeClassifier, tuned_parameters, cv=n_folds, refit=True)
clf.fit(X_train, y_train)
params = clf.best_params_
score = clf.best_score_
print("____________________________________________")
print("DecisionTreeClassifier best parameters:", params)
print("DecisionTreeClassifier score:", score)

predictions_np[:,1] = clf.predict(X_train)

with open('settings.py', 'a') as f:
	f.write('\nDecisionTreeClassifier:')
	f.write(' ' + str(params))
	f.write(' ' + str(score))

#Optimize hyperparameters for AdaBoostClassifier:
AdaBoostClassifier = AdaBoostClassifier()

n_folds = 3

tuned_parameters = {'learning_rate':scipy.stats.expon(scale=0.1) }
# I use Random hyperparameter optimization:
clf = RandomizedSearchCV(AdaBoostClassifier, tuned_parameters, cv=n_folds, refit=True)
clf.fit(X_train, y_train)
params = clf.best_params_
score = clf.best_score_
print("____________________________________________")
print("AdaBoostClassifier best parameters:", params)
print("AdaBoostClassifier score:", score)

predictions_np[:,2] = clf.predict(X_train)

with open('settings.py', 'a') as f:
	f.write('\nAdaBoostClassifier:')
	f.write(' ' + str(params))
	f.write(' ' + str(score))

#Optimize hyperparameters for RandomForestClassifier:
RandomForestClassifier = RandomForestClassifier()

n_folds = 3

tuned_parameters = {'n_estimators': scipy.stats.randint(low=6,high=16), 'max_depth':scipy.stats.randint(low=6,high=16),'min_samples_leaf':scipy.stats.randint(low=1,high=10) }
# I use Random hyperparameter optimization:
clf = RandomizedSearchCV(RandomForestClassifier, tuned_parameters, cv=n_folds, refit=True)
clf.fit(X_train, y_train)
params = clf.best_params_
score = clf.best_score_
print("____________________________________________")
print("RandomForestClassifier best parameters:", params)
print("RandomForestClassifier score:", score)

predictions_np[:,3] = clf.predict(X_train)

with open('settings.py', 'a') as f:
	f.write('\nRandomForestClassifier:')
	f.write(' ' + str(params))
	f.write(' ' + str(score))


with open('settings.py', 'a') as f:
	f.write('\nBye!!!!')

print(predictions_np)

cols = [ 'DecisionTreeClassifier', 'AdaBoostClassifier', 'RandomForestClassifier']
predictions_DF = pd.DataFrame(predictions_np, columns = cols)
print(predictions_DF.head())
print(predictions_DF.corr())

##Need to think of a way to combine the results

predictions_DF['av_predict'] = predictions_DF.sum(axis=1)
print(predictions_DF['av_predict'])