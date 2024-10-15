#################################################################
#
#  ML - HW 1
#
#  Martine De Cock
#
#################################################################

import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# Read the dataset into a dataframe and map the labels to numbers
df = pd.read_csv('iris.csv')
map_to_int = {'setosa':0, 'versicolor':1, 'virginica':2}
df["label"] = df["species"].replace(map_to_int)
print(df)

# Separate the input features from the label
features = list(df.columns[:4])
X = df[features]
y = df["label"]

# Train a decision tree and compute its training accuracy
clf = tree.DecisionTreeClassifier(max_depth=2, criterion='entropy')
clf.fit(X, y)
print(metrics.accuracy_score(y,clf.predict(X)))

# Perform 10-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

# Print the average accuracy across all folds
print(f"cross-validation score: \n{cv_scores}")
print("10-fold cross-validation accuracy: {:.2f}".format(cv_scores.mean()))

# result
# cross-validation score: 
# [0.93333333 0.93333333 1.         0.93333333 0.93333333 0.86666667
#  0.86666667 1.         1.         1.        ]
# 10-fold cross-validation accuracy: 0.95