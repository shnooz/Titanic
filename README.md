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

## Usage
+ Run python process.py to process the train and test data. 
++ That will create two new files in the processed folder.
+ Run python learn.py.
++ This file runs three different classifying algorithms and assesses them.
+ Run python predict.py.
++ This file uses logistic regression algorithm to predict the test dataset survival.
++ It will creat a new file test_predict in the predictions folder

Enjoy!
Shnooz