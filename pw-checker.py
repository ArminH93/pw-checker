import getpass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the data
data = pd.read_csv('data.csv', on_bad_lines='skip')
data = data.dropna()  # drop rows with missing values

# Rename column values from 0,1,2 to weak, medium, strong
data['strength'] = data['strength'].map({0: 'weak', 1: 'medium', 2: 'strong'})


def word(password):  # function to split the password into characters
    character = []
    for i in password:
        character.append(i)
    return character


# convert password and strength columns to arrays
pw = np.array(data['pw'])
strength = np.array(data['strength'])

# TfidVectorizer object to convert the password into a vector
tfid = TfidfVectorizer(analyzer=word)
pw = tfid.fit_transform(pw) # fit the data to the object

# split the data into training and testing data
pwtrain, pwtest, strengthtrain, strengthtest = train_test_split(
    pw, strength, test_size=0.05, random_state=42)


model = RandomForestClassifier()  # create a RandomForestClassifier object
model.fit(pwtrain, strengthtrain)  # fit the data to the object
print(model.score(pwtest, strengthtest))  # print the accuracy of the model

user = getpass.getpass('Enter your password: ')  # get the user's password
data = tfid.transform([user]).toarray()  # transform the user's password
output = model.predict(data)  # predict the strength of the user's password
print('The strength of your password is: ', output)  # print the strength
