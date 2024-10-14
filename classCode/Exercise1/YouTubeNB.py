# ML in Python, exercise 1.1
# date: 01/19/2022
# name: Martine De Cock
# description: Naive Bayes model for gender recognition of YouTube bloggers

import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Reading the data into a dataframe and selecting the columns we need
df = pd.read_table("YouTube-User-Text-gender-personality.tsv")
print(df.shape)


data_YouTube = df.loc[:,['transcripts', 'gender']]

# Splitting the data into 300 training instances and 104 test instances
n = 104
all_Ids = np.arange(len(data_YouTube)) 
#random.shuffle(all_Ids)
test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
data_test = data_YouTube.loc[test_Ids, :]
data_train = data_YouTube.loc[train_Ids, :]

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['transcripts'])
y_train = data_train['gender']
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Testing the Naive Bayes model
X_test = count_vect.transform(data_test['transcripts'])
y_test = data_test['gender']
y_predicted = clf.predict(X_test)

# Reporting on classification performance
print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))
classes = ['Male','Female']
cnf_matrix = confusion_matrix(y_test,y_predicted,labels=classes)
print("Confusion matrix:")
print(cnf_matrix)
