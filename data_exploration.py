import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

def hasNans (df):
	hasNan = False
	if df.count().min() == df.shape[0]:
		print ("Yay! There are no Nans in the dataset")
	else:
		print("oooops.. We have Nans...")
		hasNan = True
	return(hasNan)

train_df = pd.read_csv(os.path.join('data', 'train.csv'))
test_df = pd.read_csv(os.path.join('data', 'test.csv'))

print('_______________________________________________')
print('Here is the head of train:')
print (train_df.head())
print('Here is the head of test:')
print (test_df.head())
print('_______________________________________________')

print('______train dataset information________________')
print(train_df.dtypes)

print('_______________________________________________')
print("Checking for Nans in the train and test datasets:")
if hasNans(train_df) == True or hasNans(test_df) == True:
	nas = pd.concat([train_df.isnull().sum(), test_df.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset'])
	print("Here is the Nans count:")
	print(nas[nas.sum(axis=1)>0])	
print("_______________________________________________")
print("Analyzing different features impact on survival:")
print("Pcalss:")
print("_______")
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("Sex:")
print("_______")
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("SibSp:")
print("_______")
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("Parch:")
print("_______")
print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("Embarked:")
print("_______")
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))