import os
import pandas as pd

def dummify(df, col_name):
	df[col_name] = pd.get_dummies(df[col_name])
	return df


if __name__ == "__main__":
	#Reading the files into a DataFrame:
    train_df = pd.read_csv(os.path.join('data', 'train.csv'))
    test_df = pd.read_csv(os.path.join('data', 'test.csv'))

    # Changing to Categorical cols (dummy variables):
    for df in [train_df, test_df]:
    	for col_name in ['Sex', 'Embarked']:
    		df = dummify (df, col_name)

    #Choosing features to work with:
    train_df = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Survived' ]]
    test_df = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]


    #Saving the processed DataFrames to_csv in the processed folder
    train_df.to_csv(os.path.join('processed', "train.csv"))
    test_df.to_csv(os.path.join('processed', "test.csv"))