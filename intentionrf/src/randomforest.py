#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import ast
from numpy.linalg import inv
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

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

def evaluate_model(predictions, probs, train_predictions, train_probs, test_labels, train_labels):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    baseline = {}
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    for metric in ['recall', 'precision', 'roc']:
        print({metric.capitalize()}, 'Baseline: ',  {round(baseline[metric], 2)}, 'Test: ',
              {round(results[metric], 2)}, 'Train: ', {round(train_results[metric], 2)})
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve');
    plt.savefig('roc.png')
################################# DATA SPLITTING #####################

X_temp = createOneDataFrame()
X = X_temp.loc[:, X_temp.columns != 'label']
y = X_temp.iloc[:, -1]

# add mirrored y_values
X['inv_pos_y'] = mirror(X)[0]
X['inv_vel_y'] = mirror(X)[1]

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=0.2, random_state=42)

############# Create Random Forest #################
forest = RandomForestClassifier(n_estimators=100, random_state=23, max_features='sqrt')
forest.fit(X_train, y_train)

# Training predictions
train_rf_pred = forest.predict(X_train)
print('Training Accuracy:', metrics.accuracy_score(y_train, train_rf_pred))
train_rf_probs = forest.predict_proba(X_train)[:, 1]

rf_pred = forest.predict(X_test)
print('Test Mean absolute Error:', metrics.mean_absolute_error(y_test, rf_pred))
print('Test Accuracy:', metrics.accuracy_score(y_test, rf_pred))
rf_probs = forest.predict_proba(X_test)[:, 1]

evaluate_model(rf_pred, rf_probs, train_rf_pred, train_rf_probs, y_test, y_train)

# # Print Out of Bag Error (Mean Prediction Error, Error of unseen bootstrapped data) number of correct prediced rows from out of bag sample
#print('OOB-score', forest.oob_score_)


#### FEATURE IMPORTANCES ###########

features = list(X_train.columns)
fi_model = pd.DataFrame({'feature': features,'importance': forest.feature_importances_}).sort_values('importance', ascending = False)
print(fi_model)


############# PLOTS ############################

########### PLOT CONFUSION MATRIX ####################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

# Confusion matrix
cm = confusion_matrix(y_test, rf_pred)
plot_confusion_matrix(cm, classes = ['Passing', 'Interaction'],
                      title = 'Intention Confusion Matrix')

plt.savefig('cm_test.png')


############# PLOT TREE OF FOREST ###############

# Export a tree from the forest
# export_graphviz(forest, rounded = True,
#                 feature_names=X_train.columns, max_depth = 8,
#                 class_names = ['poverty', 'no poverty'], filled = True)

# call(['dot', '-Tpng', 'tree_from_optimized_forest.dot', '-o', 'tree_from_optimized_forest.png', '-Gdpi=200'])
# Image('tree_from_optimized_forest.png')

