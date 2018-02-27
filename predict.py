import os
import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def replace_nans(test):
	# Create our imputer to replace missing values with the mean e.g.
	values = {}
	for col in test.columns:
		values[col] = test[col].mean()

	test = test.fillna(value = values, axis = 0 )
	return(test)

def fit_model(train, clf):	
	predictors = train.columns.tolist()
	predictors = [p for p in predictors if p != "Survived"]
    #Run cross validation across the train DataFrame.
	model = clf.fit(train[predictors], train['Survived'])
	return model

def make_predictions (test, model):
	predictions = model.predict(test)
	return predictions

    
if __name__ == "__main__":
	clf = LogisticRegression(random_state=1, class_weight="balanced")
	print ('Reading the train dataset......')
	train = pd.read_csv(os.path.join('processed', "train.csv"))
	print('Replace Nans in the train dataset........')
	train = replace_nans(train)
	print ('fitting the model to the train dataset..........')
	model = fit_model(train, clf)



	print ('Reading the test dataset.......')
	test = pd.read_csv(os.path.join('processed', "test.csv"))
	print('Replace Nans in the test dataset........')
	test = replace_nans(test)
	#compute predictions 
	print('------------------------------------------------------------------------------')
	print ('Computing predictions for the test dataset with LOGISTIC REGRESSION......')
	print('------------------------------------------------------------------------------')
	#add prediction_stacking from different models
	test['predictions'] = make_predictions(test, model)
	print (test.head())
	test_kaggle = pd.read_csv(os.path.join('data', 'test.csv'),  usecols = ['PassengerId'])
	test_kaggle['Survived'] = test['predictions']
	test_kaggle = test_kaggle.set_index('PassengerId')
	test.to_csv(os.path.join('predictions', "test_predict.csv"))
	test_kaggle.to_csv(os.path.join('predictions', "test_predict_kaggle.csv"))
