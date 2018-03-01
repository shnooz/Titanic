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
#Importing feature_selection:
from sklearn.feature_selection import SelectFromModel
#For stacking results from different classifiers:
from sklearn.ensemble import VotingClassifier

train = pd.read_csv(os.path.join('processed', "train.csv"))

X_train = train.copy()
X_train.drop('Survived', axis=1, inplace=True)
print(X_train.head())
y_train = train['Survived']
#cols = train.columns.to_list()
#predictors = [c for c in cols if c != 'Survived']




m = X_train.shape[0]
n = X_train.shape[1]
number_of_estimators = 3 #I'm using 3 different classifiers: logisticRegrassion, AdaBoost, DecisionTree, RandomForests

#Create a numpy array to store the estimators best predictions
predictions_np = np.zeros([m,number_of_estimators])

#Optimize hyperparameters for DecisionTreeClassifier:
DecisionTreeClassifier = DecisionTreeClassifier()

n_folds = 3

tuned_parameters = {'max_depth':scipy.stats.randint(low=6,high=16),'min_samples_leaf':scipy.stats.randint(low=1,high=10) }
#feature selection:
DTC = DecisionTreeClassifier.fit(X_train,y_train)

model = SelectFromModel(DTC, prefit=True)
X_DTC = model.transform(X_train)
# I use Random hyperparameter optimization:
clf1 = RandomizedSearchCV(DecisionTreeClassifier, tuned_parameters, cv=n_folds, refit=True, n_iter=50)
clf1.fit(X_DTC, y_train)
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

tuned_parameters = {'learning_rate':scipy.stats.expon(scale=10) }
#feature selection:
ABC = AdaBoostClassifier.fit(X_train,y_train)

model = SelectFromModel(ABC, prefit=True)
X_ABC = model.transform(X_train)

# I use Random hyperparameter optimization:
clf2 = RandomizedSearchCV(AdaBoostClassifier, tuned_parameters, cv=n_folds, refit=True,  n_iter=10)
clf2.fit(X_ABC, y_train)
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
#feature selection:
RFC = RandomForestClassifier.fit(X_train,y_train)

model = SelectFromModel(RFC, prefit=True)
X_RFC = model.transform(X_train)

# I use Random hyperparameter optimization:
clf3 = RandomizedSearchCV(RandomForestClassifier, tuned_parameters, cv=n_folds, refit=True, n_iter=10)
clf3.fit(X_RFC, y_train)
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

#Optimize hyperparameters for LogisticRegressionClassifier:
LogisticRegression = LogisticRegression()

n_folds = 3

tuned_parameters = {'C': scipy.stats.expon(1)}
#feature selection:
LR = LogisticRegression.fit(X_train,y_train)

model = SelectFromModel(LR, prefit=True)
X_LR = model.transform(X_train)

# I use Random hyperparameter optimization:
clf4 = RandomizedSearchCV(LogisticRegression, tuned_parameters, cv=n_folds, refit=True, n_iter=10)
clf4.fit(X_LR, y_train)
params_LGC = clf4.best_params_
score = clf4.best_score_
print("____________________________________________")
print("LogisticRegressionClassifier best parameters:", params_LGC)
print("LogisticRegressionClassifier score:", score)


with open('settings.py', 'a') as f:
	f.write('\nLogisticRegression:')
	f.write(' ' + str(params_LGC))
	f.write(' ' + str(score))

with open('settings.py', 'a') as f:
	f.write('\nBye!!!!')

#Now lets stack the results from the optimized classifiers:

#First of all let's see how correlated the algorithms predictions are:
results = pd.DataFrame()
results['DTC'] = clf1.predict(X_DTC)
results['ABC'] = clf2.predict(X_ABC)
results['RFC'] = clf3.predict(X_RFC)
results['LG'] = clf4.predict(X_LR)

print(results.corr())

algo_predictions = pd.DataFrame([])

#eclf = VotingClassifier(estimators=[('dt', clf1),('ab',clf2) , ('rf', clf3),('lr', clf4)], voting='hard', weights=[1,1,2,1])

eclf = VotingClassifier(estimators=[('dt', clf1),('lg',clf4) , ('rf', clf3)], voting='hard',)


eclf = eclf.fit(X_train,y_train)
print("____________________________________________")
print("VotingClassifier score:")
print(eclf.score(X_train,y_train))
print("____________________________________________")


#OK! and now - let's predict the train dataset survival:
test = pd.read_csv(os.path.join('processed', 'test.csv'))
X_test = test
test['predictions'] = eclf.predict(X_test)
test_kaggle = pd.read_csv(os.path.join('data', 'test.csv'),  usecols = ['PassengerId'])
test_kaggle['Survived'] = test['predictions']
test_kaggle = test_kaggle.set_index('PassengerId')

test_kaggle.to_csv(os.path.join('predictions', "test_predict_kaggle.csv"))

