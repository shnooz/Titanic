import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dummify(df, col_name):
	df[col_name] = pd.get_dummies(df[col_name])
	return df

#def normalize(df, col_name):
#	col_mean = df[col_name].mean()
#	col_std = df[col_name].std()
#	df[col_name] = (df[col_name] - col_mean)/(col_std)**2
#	return df

def replace_nans(df, col_name):
	# Create our imputer to replace missing values with the mean e.g.
	values = {}
	#for col in df.columns:
	#	values[col] = df[col].mean()
	mean = df[col_name].mean()

	df[col_name] = df[col_name].fillna(value = mean, axis = 0 )
	return(df)

def hasNans (df):
	hasNan = False
	if df.count().min() == df.shape[0]:
		print ("Yay! There are no Nans in the dataset")
	else:
		print("oooops.. We have Nans...")
		hasNan = True
	return(hasNan)

if __name__ == "__main__":
	#Reading the files into a DataFrame:
    train_df = pd.read_csv(os.path.join('data', 'train.csv'))
    test_df = pd.read_csv(os.path.join('data', 'test.csv'))

    # Changing to Categorical cols (dummy variables):
    for df in [train_df, test_df]:
    	for col_name in ['Sex', 'Embarked']:
    		df = dummify (df, col_name)

    # Normalizing parametes:
    #for df in [train_df, test_df]:
    #	for col_name in ['Age', 'SibSp', 'Parch', 'Fare']:
    #		df = normalize(df, col_name)

    #Choosing features to work with:
    train_df = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Survived' ]]
    test_df = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

    #Replacing Nans with mean (could improve that...):
    for df in [train_df, test_df]:
    	for col_name in ['Age', 'Embarked']:
    		df = replace_nans(df, col_name)

    for df in [train_df, test_df]:
    	print(df.head())
    	print('_______')
    	hasNans (df)

    #Saving the processed DataFrames to_csv in the processed folder
    train_df.to_csv(os.path.join('processed', "train.csv"))
    test_df.to_csv(os.path.join('processed', "test.csv"))

#Producing a Heatmap to show correlation between the different fetures...
print('Producing a Heatmap to show correlation between the different fetures...')
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
print('Producing a pairplot of the features....')
g = sns.pairplot(train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Parch', 'Embarked']], hue='Survived', palette='seismic', size=1.2, diag_kind='kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
g.set(xticklabels=[])
#plt.plot(train_df['Pclass'], train_df['Survived'])
plt.show()