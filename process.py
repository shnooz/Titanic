import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dummify(df, col_name):
	dummies = pd.get_dummies(df[col_name], prefix =col_name )
	df = pd.concat([df,dummies],axis=1)
	df.drop(col_name, axis=1, inplace=True)
	return df

#def normalize(df, col_name):
#	col_mean = df[col_name].mean()
#	col_std = df[col_name].std()
#	df[col_name] = (df[col_name] - col_mean)/(col_std)**2
#	return df

def replace_nans(df, col_name):
	# Create our imputer to replace missing values with the mean e.g.

	if type(df[col_name][0]) == int or type(df[col_name][0]) == float:
		mean = df[col_name].mean()
			#df[col_name] = df[col_name].fillna(value = mean, axis = 0 )
			#return(df)
	else:
		mean = df[col_name].value_counts(dropna=True).argmax()
			
		 
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

def get_titles(df):
    
    # we extract the title from each name
    df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    df['Title'] = df.Title.map(Title_Dictionary)
    df.drop('Name', axis=1, inplace=True)
    return(df)


#Reading the files into a DataFrame:
train_df = pd.read_csv(os.path.join('data', 'train.csv'))
test_df = pd.read_csv(os.path.join('data', 'test.csv'))

    
#Using the titles from the names:
train_df = get_titles(train_df)
test_df = get_titles(test_df)


for df in [train_df, test_df]:
    for col_name in ['Age', 'Embarked','Fare']:
    	df = replace_nans(df, col_name)


	# Changing to Categorical cols (dummy variables):
for col_name in ['Sex', 'Embarked', 'Title', 'Pclass' ]:
	train_df = dummify(train_df, col_name)
	test_df = dummify(test_df,col_name)

    
    # Normalizing parametes:
    #for df in [train_df, test_df]:
    #	for col_name in ['Age', 'SibSp', 'Parch', 'Fare']:
    #		df = normalize(df, col_name)

#Deleting un-needed fetures from the DataFrames:
for col_name in ['PassengerId', 'Ticket', 'Cabin', 'Sex_male']:
	train_df.drop(col_name, axis=1, inplace=True)
	test_df.drop(col_name, axis=1, inplace=True)


print(train_df.info())
print(test_df.info())
#train_df = train_df[['Pclass', 'Sex_female', 'Age','Fare', 'SibSp', 'Parch', 'Embarked_C', 'Embarked_S', 'Title', 'Survived' ]]
#test_df = test_df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked', 'Title']]

    #Replacing Nans with mean (could improve that...):

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
#print('Producing a pairplot of the features....')
#g = sns.pairplot(train_df, hue='Survived', palette='seismic', size=1.2, diag_kind='kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
#g.set(xticklabels=[])
#plt.show()