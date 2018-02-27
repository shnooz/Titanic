# Titanic: Machine Learning from Disaster | Kaggle

![](https://wallpapercave.com/wp/haWUUOd.jpg?raw=true)

 As written on the kaggle webpage: https://www.kaggle.com/c/titanic :
 > In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

 So - that's what I'm going to do!

# Installation
## Download the data
+ Clone this repo to your computer.
+ Get into the folder using cd Titanic.
+ Make sure there are two files in your data folder (train and test)
++ You can also find the data here: https://www.kaggle.com/c/titanic/data
+ Delete the content of the processed folder and the predictions folder (so that you can create the files by running the code... )

## Install the requirements
+ Install the requirements using pip install -r requirements.txt. 
++ Make sure you use Python 2.
++ You may want to use a virtual environment for this.

## Workflow
+ Here is an outline of the workflow I used:
![](https://github.com/shnooz/Titanic/blob/master/process.png)
![](https://github.com/shnooz/Titanic/blob/master/process2.png)

## Usage
+ Run pythn data_exploration.py to explore the data. This file will output information on the data. You don't have torun it if you don't want to...
+ Run python process.py to process the train and test data. 
++ That will create two new files in the processed folder.
+ Run python optimize.py.
++ This file does several things:
+++ Random hyperparamater tuning for three different classifiers (DecisionTree, AdaBoost, RandomForests).
+++ Outputs each classifier score
+++ stack the three classifiers results (VotingClassifier with 'hard' voting).
+++ make predictions on the test.csv and saves the predictions in the predictions folder.

Enjoy!
Shnooz