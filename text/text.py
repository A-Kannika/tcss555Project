# --------------------------------------------
# Script Name: text.py
# Author: Kannika Armstrong
# Date: October 14, 2024
# Description: 
#   The script for the TCSS555 Machine Learning Project.
#   to predict the user info from text/comments.
# --------------------------------------------

import csv
import os
import codecs
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


##### Main Method #####
def main():
    prepare_training_data()
    prepare_public_data()
    # predict_gender_NB()
    write_predicted_to_table()
    

##### Write the prediction gender data into the profile.csv in public-test-data
def write_predicted_to_table():
    y_gender_predicted = predict_gender_NB()
    # print(y_gender_predicted)
    # print(len(y_gender_predicted))

    # Load the profile.csv under data/public-test-data/profile/ file into a DataFrame
    df = pd.read_csv("../data/public-test-data/profile/profile.csv")

    # Add new data to a specific column, ensuring the length matches the DataFrame's number of rows
    df['gender'] = y_gender_predicted

    # Save the updated DataFrame back to the CSV file
    df.to_csv("../data/public-test-data/profile/profile.csv", index=False)


##### Naive Bayes model for gender recognition #####
def predict_gender_NB():
    # Reading the data into a dataframe and selecting the columns we need
    df = pd.read_csv("training_list.csv")
    # print(df.shape)

    data_Facebook = df.loc[:,['gender', 'text']]
    # print(data_Facebook)

    # Splitting the data into 8000 training instances and 1500 test instances
    n = 1500
    all_Ids = np.arange(len(data_Facebook))
    # print(all_Ids)
    # random.shuffle(all_Ids)
    test_Ids = all_Ids[0:n]
    train_Ids = all_Ids[n:]
    data_test = data_Facebook.loc[test_Ids, :]
    data_train = data_Facebook.loc[train_Ids, :]

    # Training a Naive Bayes model
    count_vect = CountVectorizer()
    X_train = count_vect.fit_transform(data_train['text'])
    y_train = data_train['gender']
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Testing the Naive Bayes model
    X_test = count_vect.transform(data_test['text'])
    # print(X_test)
    y_test = data_test['gender']
    y_test_predicted = clf.predict(X_test)
    # print(len(y_predicted))
    # print(y_predicted)
    # Reporting on classification performance
    print("Accuracy: %.2f" % accuracy_score(y_test,y_test_predicted))
    classes = ["male","female"]
    cnf_matrix = confusion_matrix(y_test,y_test_predicted,labels=classes)
    print("Confusion matrix:")
    print(cnf_matrix)

    # Predicting the data From Naive Bayes model
    # Reading data from the public list
    df2 = pd.read_csv("public_data_list.csv")
    public_data_Facebook = df2.loc[:,['gender', 'text']]
    X_public = count_vect.transform(public_data_Facebook['text'])
    y_gender_predicted = clf.predict(X_public)
    # print(y_gender_predicted)
    # print(len(y_gender_predicted))

    # return the list of result
    return y_gender_predicted

##### Method to Prepare the training data by adding the text from text files to the profele.csv #####
def prepare_training_data():
    # Arrays for training data
    # Create empty array for userid, age, gender and five persionality - openness(ope), conscientiousness(con)
    # extroversion(ext), agreeableness(agr), emotional stability(neu)
    userIDArr = []
    ageArr = []
    genderArr = []
    opeArr = []
    conArr = []
    extArr = []
    agrArr = []
    neuArr = []

    # Create Emptyarray for userid and content in text file corresponding with userid in the training data.
    textIDArr = []
    textContentArr = []

    # size of training list table
    w_training_table = 9
    h_training_table = 9500

    # Create the training list as a 2D array
    # Initialize an empty list
    training_list = []

    # Create the empty table and add 0 to all cells to prepare the table
    for y in range(h_training_table):
        # Create a row of zeros
        row = [0] * w_training_table
        # Append the row to the training_list
        training_list.append(row)

    # size of text files table
    w_text_files = 2
    h_text_files = 9500

    # Initialize an empty list
    text_list = []

    # Create the empty table and add 0 to all cells to prepare the table
    for y in range(h_text_files):
        # Create a row of zeros
        row = [0] * w_text_files
        # Append the row to text_list
        text_list.append(row)
    
    # the header is use for output csv file
    header_training = ['userid', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu', 'text']

    # path for the directory of profile.csv
    path_profile_csv = open('../data/training/profile/profile.csv')

    #path for the text files of the training data
    path_training_data_text = '../data/training/text/'


    # open training data csv file 
    with path_profile_csv as csv_file:
        read_csv = csv.reader(csv_file, delimiter = ',')

        # skip the first row
        next(csv_file)

        # row count is to count the total of training target
        row_count = 0

        # Get all training  target info into arrays by using for loop
        # and increment the row_count variable
        for row in read_csv:
            userid = row[1]
            age = row[2]
            gender = row[3]
            ope = row[4]
            con = row[5]
            ext = row[6]
            agr = row[7]
            neu = row[8]

            userIDArr.append(userid)
            ageArr.append(age)
            genderArr.append(gender)
            opeArr.append(ope)
            conArr.append(con)
            extArr.append(ext)
            agrArr.append(agr)
            neuArr.append(neu)
            row_count += 1

        # total amount of text files
        row_count_txt_file = 0

        # Use a for loop to insert data into the training data list.
        for row in range(row_count):
            training_list[row][0] = userIDArr[row]
            training_list[row][1] = "%.0f" % float(ageArr[row])
            training_list[row][2] = "%.0f" % float(genderArr[row])
            training_list[row][3] = opeArr[row]
            training_list[row][4] = conArr[row]
            training_list[row][5] = extArr[row]
            training_list[row][6] = agrArr[row]
            training_list[row][7] = neuArr[row]
            row_count_txt_file += 1

        # Convert age number to age range
        # xx-24 = 1; 25-34 = 2; 35-49 = 3; 50-xx = 4;
        # Convert gender number to text
        # 0 = "male", 1 = "female"

        for row in range(len(training_list)):
            for col in range(7):
                if(col == 1):
                    if int(training_list[row][1]) <= 24:
                        training_list[row][1] = 1
                    elif 24 < int(training_list[row][1]) <= 34:
                        training_list[row][1] = 2
                    elif 34 < int(training_list[row][1]) <= 49:
                        training_list[row][1] = 3
                    else:
                        training_list[row][1] = 4
                if(col == 2):
                    if int(training_list[row][2]) == 0:
                        training_list[row][2] = "male"
                    else:
                        training_list[row][2] = "female"
        
        # Use for loop to go over every files in text folder
        for filename in os.listdir(path_training_data_text):   

            # create content variable
            content = ""

            # There is an user id in every text file
            # filename[:-4] is get rid of last four characters for the text file name (.txt)
            textIDArr.append(filename[:-4])

            # UUse the codecs module to handle UnicodeDecodeError in some text files. 
            # The key to solving this issue is setting errors='ignore'.
            contents = codecs.open(path_training_data_text + filename, encoding='utf-8', errors='ignore')

            # Go over content by line
            for line in contents:

                # combine all into the content variable
                content += line

            # Append content in text file to an array
            textContentArr.append(content)

        # Add textIDArr and textContentArr into text_list
        for row in range(row_count_txt_file):
            text_list[row][0] = textIDArr[row]
            text_list[row][1] = textContentArr[row]

        # Compare user ID in training_list and user ID in text_list
        training_list.sort(key=lambda elem: elem[0])
        text_list.sort(key=lambda elem: elem[0])
        for row in range(row_count):
            training_list[row][8] = text_list[row][1]
        
    # create the training list for our model: training_list.csv
    with codecs.open('training_list.csv', 'w', encoding='utf-8', errors='ignore') as csv_output_file:
        write_csv = csv.writer(csv_output_file, delimiter = ',')
        write_csv.writerow(header_training)
        write_csv.writerows(training_list)
        
    # close all files.
    csv_file.close()
    contents.close()
    csv_output_file.close()

##### Method to Prepare the public data by adding the text from text files to the profele.csv #####
def prepare_public_data():
    # Arrays for data we want to predict
    # Create empty array for userid, age, gender and five persionality - openness(ope), conscientiousness(con)
    # extroversion(ext), agreeableness(agr), emotional stability(neu)
    userIDArr = []
    ageArr = []
    genderArr = []
    opeArr = []
    conArr = []
    extArr = []
    agrArr = []
    neuArr = []

    # Create Emptyarray for userid and content in text file corresponding with userid in the training data.
    textIDArr = []
    textContentArr = []

    # size of training list table
    w_table = 9
    h_table = 334

    # Create the training list as a 2D array
    # Initialize an empty list
    public_data_list = []

    # Create the empty table and add 0 to all cells to prepare the table
    for y in range(h_table):
        # Create a row of zeros
        row = [0] * w_table
        # Append the row to the training_list
        public_data_list.append(row)

    # size of text files table
    w_text_files = 2
    h_text_files = 334

    # Initialize an empty list
    text_list = []

    # Create the empty table and add 0 to all cells to prepare the table
    for y in range(h_text_files):
        # Create a row of zeros
        row = [0] * w_text_files
        # Append the row to text_list
        text_list.append(row)
    
    # the header is use for output csv file
    header_table = ['userid', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu', 'text']

    # path for the directory of profile.csv
    path_profile_csv = open('../data/public-test-data/profile/profile.csv')

    # path for the text files of the public-test-data data
    path_public_data_text = '../data/public-test-data/text/'

    # open training data csv file 
    with path_profile_csv as csv_file:
        read_csv = csv.reader(csv_file, delimiter = ',')

        # skip the first row
        next(csv_file)

        # row count is to count the total of training target
        row_count = 0

        # Get all training  target info into arrays by using for loop
        # and increment the row_count variable
        for row in read_csv:
            userid = row[1]
            age = row[2]
            gender = row[3]
            ope = row[4]
            con = row[5]
            ext = row[6]
            agr = row[7]
            neu = row[8]

            userIDArr.append(userid)
            ageArr.append(age)
            genderArr.append(gender)
            opeArr.append(ope)
            conArr.append(con)
            extArr.append(ext)
            agrArr.append(agr)
            neuArr.append(neu)
            row_count += 1

        # total amount of text files
        row_count_txt_file = 0

        # Use a for loop to insert data into the training data list.
        for row in range(row_count):
            public_data_list[row][0] = userIDArr[row]
            public_data_list[row][1] = ageArr[row]
            public_data_list[row][2] = genderArr[row]
            public_data_list[row][3] = opeArr[row]
            public_data_list[row][4] = conArr[row]
            public_data_list[row][5] = extArr[row]
            public_data_list[row][6] = agrArr[row]
            public_data_list[row][7] = neuArr[row]
            row_count_txt_file += 1
        # print(public_data_list[0])

        

        # Use for loop to go over every files in text folder
        for filename in os.listdir(path_public_data_text):   

            # create content variable
            content = ""

            # There is an user id in every text file
            # filename[:-4] is get rid of last four characters for the text file name (.txt)
            textIDArr.append(filename[:-4])

            # UUse the codecs module to handle UnicodeDecodeError in some text files. 
            # The key to solving this issue is setting errors='ignore'.
            contents = codecs.open(path_public_data_text + filename, encoding='utf-8', errors='ignore')

            # Go over content by line
            for line in contents:

                # combine all into the content variable
                content += line

            # Append content in text file to an array
            textContentArr.append(content)
        
        # Add textIDArr and textContentArr into text_list
        for row in range(row_count_txt_file):
            text_list[row][0] = textIDArr[row]
            text_list[row][1] = textContentArr[row]

        # Compare user ID in training_list and user ID in text_list
        public_data_list.sort(key=lambda elem: (str(elem[0]) if isinstance(elem[0], int) else elem[0]))
        text_list.sort(key=lambda elem: (str(elem[0]) if isinstance(elem[0], int) else elem[0]))

        for row in range(row_count):
            public_data_list[row][8] = text_list[row][1]
        
    # create the training list for our model: training_list.csv
    with codecs.open('public_data_list.csv', 'w', encoding='utf-8', errors='ignore') as csv_output_file:
        write_csv = csv.writer(csv_output_file, delimiter = ',')
        write_csv.writerow(header_table)
        write_csv.writerows(public_data_list)
        
    # close all files.
    csv_file.close()
    contents.close()
    csv_output_file.close()

if __name__=="__main__":
    main()

