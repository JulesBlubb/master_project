#!/usr/bin/env python

from matplotlib.legend_handler import HandlerLine2D
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, mean_squared_error, confusion_matrix, auc, accuracy_score, classification_report
from spicy import interp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from sklearn.tree import export_graphviz
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from randomforest import splitting


X_train, y_train, X_test, y_test, X_val, y_val = splitting()

# def plot_position():
#     X = createDf('/home/juliane/catkin_ws/src/intentionrf/src/transformed/passing_2019-07-03-13-55-00.txt', 0)
#     time= X.index.values
#     fig = px.line(X, y='pos_x')
#     fig.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

def calc_metrics(clf, train_results,test_results, X_train, y_train):
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    test_pred = clf.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, test_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
    return train_results, test_results


# #Plot Max_depth
def plot_max_depth(X_train, y_train):
    max_depths = np.linspace(1, 50, 50, endpoint=True)
    train_results = []
    test_results = []

    for max_depth in max_depths:
        clf = RandomForestClassifier(max_depth=max_depth, n_estimators=742, min_samples_split=13 ,n_jobs=-1)
        train_results, test_results = calc_metrics(clf, train_results, test_results, X_train, y_train)

    plt.figure(0)
    line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.show()

plot_max_depth(X_train, y_train)

#Plot n_estimators
def plot_nestimators(X_train, y_train):
    n_estimators = [1, 200, 500, 700, 900, 1100]
    train_results = []
    test_results = []
    for e in n_estimators:
        clf = RandomForestClassifier(max_depth=15, n_estimators=e, min_samples_split=13, n_jobs=-1)

        train_results, test_results = calc_metrics(clf, train_results, test_results, X_train, y_train)

    plt.figure(1)
    line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
    line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylim(0.4, 1.0)
    plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.show()

plot_nestimators(X_train, y_train)

#Plot min_samples_splits
def plot_minsamplessplits(X_train, y_train):
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    train_results = []
    test_results = []
    for s in min_samples_splits:
        clf = RandomForestClassifier(max_depth=15, n_estimators=300, min_samples_split=s, n_jobs=-1)
        train_results, test_results = calc_metrics(clf, train_results, test_results, X_train, y_train)

    plt.figure(2)
    line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
    line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylim(0.4, 1.0)
    plt.ylabel('AUC score')
    plt.xlabel('minimum samples per split')
    plt.show()

plot_minsamplessplits(X_train, y_train)

def plot_roc_helper(mean_fpr, aucs, tprs):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def roc_plot():
    # # #Stratification is the process of rearranging the data as to ensure each fold is a good representative of the whole.
    cv = StratifiedKFold(n_splits=10)
    model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=0.1, oob_score=True, random_state=42)
    i = 0
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    X_plot = pd.concat([X_train, X_val])
    y_plot = pd.concat([y_train, y_val])

    for train, val in cv.split(X_plot, y_plot):
        fit = model.fit(X_plot.iloc[train], y_plot.iloc[train])
        probas_ = fit.predict_proba(X_plot.iloc[val])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_plot.iloc[val], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        cm = confusion_matrix(y_plot.iloc[val], model.predict(X_plot.iloc[val]))
        i += 1

    pred = fit.predict(X_test)
    pred_train = fit.predict(X_train)
    acc = accuracy_score(y_test, pred)
    acc_train = accuracy_score(y_train, pred_train)

    print('ACC_test', acc)
    print('ACC_train', acc_train)
    print(classification_report(y_test, pred))

    print('OOB', fit.oob_score_)
    print('Score: ', model.score(X_train, y_train))
    # # Confusion matrix
    #plot_confusion_matrix(roc_plot(), classes = ['Passing', 'Interaction'],
    #                          title = 'Intention Confusion Matrix')

    plot_roc_helper(mean_fpr, aucs, tprs)
    return fit, cm, model


fit, cm, model = roc_plot()


def plot_tree(names, features):
    # print important features
    features = list(X_train.columns)
    fit, cm, model = roc_plot()
    fi_model = pd.DataFrame({'feature': features,'importance': model.feature_importances_}).sort_values('importance', ascending = False)

    names = (map(str, (model.classes_).tolist()))

    estimator = model.estimators_[6]
    export_graphviz(estimator, out_file='tree.dot',
                    feature_names = features,
                    class_names = names,
                    rounded = True, proportion = False,
                    precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])





