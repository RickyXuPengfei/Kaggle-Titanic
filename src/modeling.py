#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:38:58 2017

@author: ricky_xu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold,cross_val_score,GridSearchCV,learning_curve


def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,scoring='f1',n_jobs=1,
                        train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):
    """
    plot the learning curve of the estimator (X_axis : different training size; Y_axis: the mean of cross validation scores)

    :param estimator: object; the classifier without fitting the training data.

    :param title: string; the title of the plot.

    :param X: array; input samples of training data.

    :param y: array; target values.

    :param ylim: list; the range of Y_axis.

    :param cv: int; cross-validation generator or an iterable.

    :param scoring: string; the metric they apply to the estimators evaluated.

    :param n_jobs: int; The number of jobs to run in parallel for both fit and predict.

    :param train_sizes:array; the different training sizes to run the estimator.

    :param verbose: int; control the message.

    :param plot: boolean; determine whether it can be plotted.


    """
    train_sizes,train_scores,test_scores=learning_curve(estimator,X,y,cv=cv,scoring=scoring,n_jobs=n_jobs,train_sizes=train_sizes,verbose=verbose)
    
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel('training sizes')
        plt.ylabel('score')
#        plt.gca().invert_yaxis()
        plt.grid()
        
        plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,
                         alpha=0.1,color='b')
        plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,
                         alpha=0.1,color='r')
        plt.plot(train_sizes,train_scores_mean,'o-',color='b',label='score in training set')
        plt.plot(train_sizes,test_scores_mean,'o-',color='r',label='score in validation set')
        
        plt.legend(loc='best')
        
    plt.show()
    
class Classifier(object):
    def __init__(self,clf,name,X_train,y_train,seed=0,params=None,scoring='f1'):
        """

        :param clf: object; the classifier without fitting the training data.

        :param name: string; the name of the classifier.

        :param X_train: array; input samples of training data.

        :param y_train: array; target values.

        :param seed: int; random seed.

        :param params: dict; the values of  parameters.

        :param scoring: string; the metric they apply to the estimators evaluated.
        """
#        params['random_state']=seed
        self.name=name
        self.clf=clf(**params)
        self.X=X_train
        self.y=y_train
        self.scoring =  scoring 
        
    def train(self):
        """
        fit the training model using model.
        """
        self.clf.fit(self.X,self.y)
    
    def predict(self,x):
        """
        Predict target values of X given a mode.

        :param x: array; test samples.

        :return: array; predicted values.
        """
        return self.clf.predict(x)
    
    def GridSearch(self,params,cv=5):
        """
        exhaustive search over specified parameter values for an estimator.
        Without training.

        :param params: dict; Dictionary with parameters names (string) as keys and lists of parameter settings to try as values

        :param cv: int; cross-validation generator or an iterable.

        """

        clf=GridSearchCV(self.clf,params,cv=cv,scoring=self.scoring,verbose=1)
        clf.fit(self.X,self.y)
        best_parms=clf.best_params_
        best_score=clf.best_score_
        best_model=clf.best_estimator_
        self.clf=best_model
        print (best_score)
        print (best_parms)
        print (best_model)
    
    def plot_learning_curve(self,cv,plt=False):
        """
        plot the learning curve of the estimator (X_axis : different training size; Y_axis: the mean of cross validation scores)

        :param cv: int; cross-validation generator or an iterable.

        :param plt: boolean; determine whether it can be plotted.

        """
        
        if plt==True:
            plot_learning_curve(self.clf,cv=cv,scoring=self.scoring,title='learning curve',X=self.X,y=self.y)
    
    def CrossValScore(self,mean=False):
        """
        Evaluate a score by cross-validation.

        :param mean: boolean; if return the mean of scores.

        :return: int; the scores.
        """
        if mean==True:
            return cross_val_score(self.clf,self.X,self.y,cv=5,scoring=self.scoring).mean()
        else:
            return cross_val_score(self.clf,self.X,self.y,cv=5,scoring=self.scoring)

    def feature_importance(self,max_features=40):
        """
        Construct the feature importance.

        :param max_features: int ; the number of features showing its importance.

        :return: DataFrame; record the importance of each feature.
        """
        if self.clf.feature_importances_.any():
            indices=np.argsort(self.clf.feature_importances_)[::-1][:max_features]
            feature_importances=self.clf.feature_importances_[indices][:40]
            Features=self.X.columns[indices][:max_features]
            pd_feature_importances=pd.DataFrame({'Features':Features,'Feature_Importances':feature_importances})
        else:
            print ("valid feature_importance")
            pd_feature_importances=None
        return pd_feature_importances
    
    def plot_feature_importance(self,plot=False):
        """
        visualize the feature importance.

        :param plot: boolean; determine whether it can be plotted.

        """

        if plot == True:
            ax = sns.barplot(x='Features', y='Feature_Importances', data=self.feature_importance())
            plt.setp(ax.get_xticklabels(), rotation=40)
            plt.show()


class Stack_Ensemble(object):
    def __init__(self, n_splits, stacker, GridParams, base_models, Grid=False, cv=5, scoring='accuracy'):
        """

        :param n_splits: int; Number of folds.

        :param stacker: meta model; the final model to predict the test data.

        :param GridParams: dict; Dictionary with parameters names (string) as keys and lists of parameter settings to try as values

        :param base_models: list; a list of base models.

        :param Grid: boolean; whether exhaustive search over specified parameter values for a stacker.

        :param cv: int; cross-validation generator or an iterable.

        :param scoring: string; the metric they apply to the estimators evaluated.
        """
        self.n_splits = n_splits
        self.stacker = stacker
        self.GridParams = GridParams
        self.base_models = base_models
        self.cv = cv
        self.scoring = scoring
        self.Grid = Grid

    def fit_predict(self, X, y, T,get_result = False):
        """
        do stacking the models and predict the test data.

        :param X: array; the input samples.

        :param y: array; target values.

        :param T: array; test data.

        :param get_result: boolean; whether do prediction.

        :return: array; the prediction of test data.
        """
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)
            
        if self.Grid == True:
            clf=GridSearchCV(self.stacker,self.GridParams,cv=self.cv,scoring=self.scoring,verbose=1)
            clf.fit(S_train,y)
            best_parms=clf.best_params_
            best_score=clf.best_score_
            best_model=clf.best_estimator_
            # print ("Stacker meta model's Best Paramaters:{0}".format(best_model))
            self.stacker=best_model
            
        results = cross_val_score(self.stacker, S_train, y, cv=self.cv, scoring=self.scoring)
        print("Stacker score: %.5f" % (results.mean()))

        pd.DataFrame(S_train).to_csv("stacking_train.csv")
        pd.DataFrame(y).to_csv("stacking_y.csv")
        pd.DataFrame(S_test).to_csv("stacking_y.csv")
        if get_result == True:
            self.stacker.fit(S_train, y)
            # res = self.stacker.predict_proba(S_test)[:,1]
            res = self.stacker.predict(S_test)
            return res
        else:
            return results.mean()