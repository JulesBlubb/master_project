#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np
import pandas as pd
import ast
from numpy.linalg import inv
import glob
import os
from sklearn.model_selection import train_test_split, cross_val_score

import matplotlib.pyplot as plt

import plotly.express as px
from scipy import interp
from bayes_opt import BayesianOptimization

# Read in Files
def readIn(fileName):
    lineList = [line.rstrip('\n') for line in open(fileName)]
    return lineList

def divideFeature(openFile, bool):
    if bool:
        pos = openFile[0::3]
        vel = openFile[1::3]
        cov = openFile[2::3]
    else:
        pos = openFile[0::2]
        vel = openFile[1::2]
        cov = []
    return pos, vel, cov

def cutOffString(feat):
    result = []
    for f in feat:
        no_keyword = f[10:]
        result.append(no_keyword)
    return result

def convertString(string_list):
    result = []
    for s in string_list:
        x = ast.literal_eval(s)
        converted = map(float, x)
        result.append(converted)
    return result

def createRaws(fileName):
    if ("_transformed" in fileName):
        temp = divideFeature(readIn(fileName), True)
        raw_cov = convertString(cutOffString(temp[2]))
    else:
        temp = divideFeature(readIn(fileName), False)
        raw_cov = []

    raw_points_pos = convertString(cutOffString(temp[0]))
    raw_points_vel = convertString(cutOffString(temp[1]))

    return raw_points_pos, raw_points_vel, raw_cov

def createDf(fileName, label):
    # temp[0] = pos, temp[1] = vel, temp[2] = cov
    temp = createRaws(fileName)
    X = pd.DataFrame(np.column_stack([temp[0], temp[1]]), columns=['pos_x', 'pos_y', 'pos_z',
                                                                   'vel_x', 'vel_y', 'vel_z'])

    #             (x)         (y)        (z)
    # (x)     #  var_x     cov_x_y     cov_x_z
    # (y)     #  cov_y_x   var_y       cov_y_z
    # (z)     #  cov_z_x   cov_z_y     var_z

    if temp[2]:
    # take only upper matrix of cov because of symmetry
        t = []
        for cov in temp[2]:
            data = {'var_px': cov[0], 'cov_p_xy': cov[1], 'cov_p_xz': cov[2],
                    'var_py': cov[4], 'cov_p_yz': cov[5], 'var_pz': cov[8]}
            t.append(data)
        t = pd.DataFrame(t)
        X = pd.concat([X, t], axis=1, sort=False)

    # add previous position and velocity values
    def add_prev(n):
        for i in range(1,4):
            X[n + '_t-' + str(i)] = X[n].shift(i)

    n = ['pos_x', 'pos_y', 'vel_x', 'vel_y']
    map(lambda x: add_prev(x), n)

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
    X_temp = pd.concat(X_temp, sort=False)
    return X_temp

# # data augmentation, invert y values to mirror direction human is coming frome
def mirror(dat):
    y_values_pos = np.matrix(dat['pos_y'])
    y_values_vel = np.matrix(dat['vel_y'])
    return(y_values_pos.I, y_values_vel.I)


#from fancyimpute import KNN

# #train_cols = list(X)

# # Use 5 nearest rows which have a feature to fill in each row's
# # missing features

# #X = pd.DataFrame(KNN(k=13).fit_transform(X))
# # # train.columns = train_cols

# # Bayesian Optimization Code based on https://www.kdnuggets.com/2019/07/xgboost-random-forest-bayesian-optimisation.html
def bayesian_optimization(dataset, function, parameters):
   X_train, y_train, X_val, y_val = dataset
   n_iterations = 5
   # gp_params = {"alpha": 1e-4}
   # kernel=Matern(nu=2.5)
   #  alpha=1e-6,normalize_y=True, n_restarts_optimizer=5,
   BO = BayesianOptimization(function, parameters)
   BO.maximize(n_iter=n_iterations)

   return BO.max

def rfc_optimization(cv_splits, X_train, y_train):
    def function(n_estimators, max_depth, min_samples_split):
        return cross_val_score(
               RandomForestClassifier(
                   n_estimators=int(max(n_estimators,0)),
                   max_depth=int(max(max_depth,1)),
                   min_samples_split=int(max(min_samples_split,2)),
                   n_jobs=-1,
                   random_state=42,
                   class_weight="balanced"),
               X=X_train,
               y=y_train,
               cv=cv_splits,
               scoring="roc_auc",
               n_jobs=-1).mean()

    parameters = {"n_estimators": (10, 1000),
                  "max_depth": (1, 150),
                  "min_samples_split": (2, 150)}
    return function, parameters

def train(X_train, y_train, X_val, y_val, function, parameters):
    dataset = (X_train, y_train, X_val, y_val)
    best_solution = bayesian_optimization(dataset, function, parameters)
    params = best_solution["params"]
    print(params)
    model = RandomForestClassifier(
             n_estimators=int(max(params["n_estimators"], 0)),
             max_depth=int(max(params["max_depth"], 1)),
             min_samples_split=int(max(params["min_samples_split"], 2)),
             n_jobs=-1,
             random_state=42,
             criterion="gini",
             oob_score=True)
    return model


def splitting():
    X_temp = createOneDataFrame()
    # add mirrored y_values
    #X_temp['inv_pos_y'] = mirror(X_temp)[0]
    #X_temp['inv_vel_y'] = mirror(X_temp)[1]

    X_nona = X_temp.fillna(-99999)

    X = X_nona.drop('label', axis=1)
    #X = X_temp.drop('vel_z', axis=1)
    y = X_temp['label']

    # # Splitting data, 64% Training,16% Validation, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    return X_train, y_train, X_test, y_test, X_val, y_val

def main():
    # get hyperparamter through Bayesian Optimization
    # gives that if max_depth is very small <10, the result is poor
    # from visualizations import roc_plot
    #fit, cm, model = roc_plot()
    X_train, y_train, X_test, y_test, X_val, y_val = splitting()
    #bayes_model = train(X_train, y_train, X_val, y_val, rfc_optimization(10, X_train, y_train)[0], rfc_optimization(10, X_train, y_train)[1])
    #print(bayes_model)


main()


    # # #Stratification is the process of rearranging the data as to ensure each fold is a good representative of the whole.

# #clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=0, min_samples_split= 500, max_features='sqrt')
# # clf.fit(X_train, y_train)
# # print(clf.score(X_val, y_val))


