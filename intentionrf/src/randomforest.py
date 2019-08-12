#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import ast
from numpy.linalg import inv
import glob
import os
# functools

# Read in Files
def readIn(fileName):
    lineList = [line.rstrip('\n') for line in open(fileName)]
    return lineList


def divideFeature(openFile):
    pos = openFile[0::2]
    vel = openFile[1::2]
    return pos, vel


def cutOffString(feat):
    result = []
    for f in feat:
        no_keyword = f[10:]
        result.append(no_keyword)
    return result


def convertString(string_list):
    result = []
    for s in string_list:
        # print(s)
        x = ast.literal_eval(s)
        converted = map(float, x)
        result.append(converted)
        # print(converted)
    return result


def createRaws(fileName):
    temp = divideFeature(readIn(fileName))
    raw_points_pos = convertString(cutOffString(temp[0]))
    raw_points_vel = convertString(cutOffString(temp[1]))
    return raw_points_pos, raw_points_vel

def createDf(fileName, label):
    temp = createRaws(fileName)
    X = pd.DataFrame(np.column_stack([temp[0], temp[1]]), columns=['pos_x', 'pos_y', 'pos_z',
                                                                   'vel_x', 'vel_y', 'vel_z'])
    X['label'] = label
    return X

def createOneDataFrame():
    path = 'transformed/'
    X_temp = []
    for filename in glob.glob(os.path.join(path, '*.txt')):
        first_fileword = os.path.basename(filename).split('_')[0]
        # Interaction gets label 1 and Passing label 0
        label = 0 if first_fileword == 'passing' else 1
        X_temp.append(createDf(filename, label))
    X_temp = pd.concat(X_temp)
    return X_temp

# data augmentation, invert y values to mirror direction human is coming frome
def mirror(dat):
    y_values_pos = np.matrix(dat['pos_y'])
    y_values_vel = np.matrix(dat['vel_y'])
    return(y_values_pos.I, y_values_vel.I)

X_temp = createOneDataFrame()
X = X_temp.loc[:, X_temp.columns != 'label']
y = X_temp.iloc[:, -1]

# add mirrored y_values
X['inv_pos_y'] = mirror(X)[0]
X['inv_vel_y'] = mirror(X)[1]


# # Create Random Forest
forest = RandomForestClassifier(n_estimators=700, random_state=0, oob_score=True)

forest.fit(X, y)

# # print('Score: ', forest.score(X, y))

# # Print Out of Bag Error (Mean Prediction Error, Error of unseen bootstrapped data) number of correct prediced rows from out of bag sample
print('OOB', forest.oob_score_)


# #predict = trained_model.predict()



 #print(clf.feature_importances_)



