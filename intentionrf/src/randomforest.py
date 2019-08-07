#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier

import rospy
import numpy as np
import rosbag
import pandas as pd
import ast
import numpy as np

# Read in Files
def readIn(fileName):
    lineList = [line.rstrip('\n') for line in open(fileName)]
    return lineList

def divideFeature(openFile):
    pos = openFile[0::2]
    vel = openFile[1::2]
    return pos,vel

def cutOffString(feat):
    result = []
    for f in feat:
        no_keyword = f[10:]
        result.append(no_keyword)
    return result


def convertString(string_list):
    result = []
    for s in string_list:
        #print(s)
        x = ast.literal_eval(s)
        converted = map(float, x)
        result.append(converted)
        #print(converted)
    return result


def createRaws(fileName):
    temp = divideFeature(readIn(fileName))
    raw_points_pos = convertString(cutOffString(temp[0]))
    raw_points_vel = convertString(cutOffString(temp[1]))
    return raw_points_pos, raw_points_vel

def createDf(fileName,label):
    temp = createRaws(fileName)
    X = pd.DataFrame(np.column_stack([temp[0], temp[1]]), columns = ['pos_x', 'pos_y', 'pos_y', 'vel_x', 'vel_y', 'vel_z'])
    X['label'] = label
    return X

button_In_34 = "transformed/button_int_2019-07-03-13-28-34.txt"
button_In_11 = "transformed/button_int_2019-07-03-13-56-11.txt"
button_In_40 = "transformed/button_int_2019-07-03-13-56-40.txt"

voice_14 = "transformed/voice_int_2019-07-03-13-57-14.txt"
voice_46 = "transformed/voice_int_2019-07-03-13-57-46.txt"

passing_39 = "transformed/passing_2019-07-03-13-55-39.txt"
passing_00 = "transformed/passing_2019-07-03-13-55-00.txt"

# Interaction gets label 1 and Passing label 0
X_temp = pd.concat([(createDf(button_In_34, 1)), (createDf(passing_00, 0)), (createDf(passing_39, 0)), (createDf(button_In_11, 1)), (createDf(button_In_40, 1)), (createDf(voice_14, 1)) ])

X = X_temp.loc[:, X_temp.columns != 'label']
y = X_temp.iloc[:,-1]


# Create Random Forest
forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, oob_score=True)

forest.fit(X,y)

print('Score: ', forest.score(X, y))

# Print Out of Bag Error (Mean Prediction Error, Error of unseen bootstrapped data) number of correct prediced rows from out of bag sample
print('OOB', forest.oob_score_)


# #predict = trained_model.predict()



#print(clf.feature_importances_)



