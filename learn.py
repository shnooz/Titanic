import os
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics
#importing classifiers:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#def cross_validate(train, clf):
#	'''This function takes a train dataset and a clf (classifier) and returns predictions by using cross validation'''
#    #creats a list of columns that we want to use to train the model on, removing 'Survived' col
#	predictors = train.columns.tolist()
#	predictors = [p for p in predictors if p != "Survived"]
#   #Run cross validation across the train DataFrame.
#	predictions = cross_validation.cross_val_predict(clf, train[predictors], train['Survived'], cv=3)
#    #Return the predictions
#	return predictions

#def compute_error(target, predictions):
#	'''Uses scikit-learn to compute a simple accuracy score 
#	(the percentage of predictions that matched the actual foreclosure_status values).'''
#	return metrics.accuracy_score(target, predictions)

#def compute_false_negatives(target, predictions):
#	'''Combines the target and the predictions into a DataFrame for convenience.
#	Finds the false negative rate.'''
#	temp_dict = {"target": target, "predictions": predictions}
#	df = pd.DataFrame.from_dict(temp_dict)
#	return float(df[(df["target"] == 1) & (df["predictions"] == 0)].shape[0]) / float(df[(df["target"] == 1)].shape[0] + 1)

#def compute_false_positives(target, predictions):
#	'''Finds the false positive rate. i.e. the number of loans that weren't foreclosed 
#	on that the model predicted would be foreclosed on.'''
#	#Combines the target and the predictions into a DataFrame for convenience.
#	temp_dict = {"target": target, "predictions": predictions}
#	df = pd.DataFrame.from_dict(temp_dict)
#	return float(df[(df["target"] == 0) & (df["predictions"] == 1)].shape[0]) / float(df[(df["target"] == 0)].shape[0] + 1)

def read(file_name):
	'''Read the dataset'''
	df = pd.read_csv(os.path.join('processed', file_name))
	return df

def compute_and_print(train, clf):
	name = str(clf)
	#print('Replace Nans in the train dataset........')
	#train = replace_nans(train)
	#compute cross validated predictions

	print ('Computing prediction with' + name + '......')
	predictions_log = cross_validate(train, clf)

    #compute the three error metrics:
	error = compute_error(train['Survived'], predictions_log)
	fn = compute_false_negatives(train['Survived'], predictions_log)
	fp = compute_false_positives(train['Survived'], predictions_log)
    #print:
	print ("--------------------------------------------------------")
	print ("Here are the results for " + name + " classifier:")
	print ("--------------------------------------------------------")
	print("Accuracy Score: {}".format(error))
	print("False Negatives: {}".format(fn))
	print("False Positives: {}".format(fp))
    
if __name__ == "__main__":
	print ('Reading the train dataset.......')
	train = read()
	# add regularization
	# add optimize_parameters(train, model)
	compute_and_print(train, (LogisticRegression(random_state=1, class_weight="balanced")))
	compute_and_print(train, (DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, class_weight="balanced")))
	compute_and_print(train, (AdaBoostClassifier(learning_rate = 0.5, n_estimators=100)))
	compute_and_print(train, (RandomForestClassifier(max_depth=6, min_samples_leaf=3)))



