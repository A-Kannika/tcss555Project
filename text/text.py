# --------------------------------------------
# Script Name: text.py
# Author: Kannika Armstrong
# Date: October 14, 2024
# Description: 
#   This script trains a decision tree classifier
#   on the dataset and computes training
#   accuracy.
# --------------------------------------------

import csv
import os
import codecs

def prepare_data():
    # Create nempty array for userid, age, gender and five persionality - openness(ope), conscientiousness(con)
    # extroversion(ext), agreeableness(agr), emotional stability(neu)
    userIDArr = []
    ageArr = []
    genderArr = []
    opeArr = []
    conArr = []
    extArr = []
    agrArr = []
    neuArr = []

    # Create Emptyarray for userid and content in text file corresponding with userid.
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
        # xx-24 = 1;
        # 25-34 = 2;
        # 35-49 = 3;
        # 50-xx = 4;
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
        
    # create the traning list for our model: training_list.csv
    with codecs.open('training_list.csv', 'w', encoding='utf-8', errors='ignore') as csv_output_file:
        write_csv = csv.writer(csv_output_file, delimiter = ',')
        write_csv.writerow(header_training)
        write_csv.writerows(training_list)
        
    # close all files.
    csv_file.close()
    contents.close()
    csv_output_file.close()

prepare_data()