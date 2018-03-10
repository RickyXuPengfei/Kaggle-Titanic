#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:56:00 2018

@author: ricky_xu
"""

from modeling import Classifier
from sklearn.ensemble import RandomForestClassifier
from Feature_Generation import y_gen
from Feature_selection import new_result

rf_parameters={
               'random_state':[i*2 for i in range(6)]
               }

rf_params = {'criterion':'gini',
             'n_estimators':100,
             'max_features':'sqrt',
             'max_depth': 10,
             'min_samples_leaf': 3,
             'random_state':5,
             'n_jobs':-1}

rfc =Classifier(RandomForestClassifier,'rfc',new_result,y_gen,seed=0,params=rf_params,scoring='accuracy')
print ("RandomForest Train")
# rfc.train()
clf_rf = rfc.clf
#rfc.GridSearch(rf_parameters)
#print (rfc.feature_importance())
#sns.barplot(x='Features',y='Feature_Importances',data=rfc.feature_importance())
#plt.show()
# print (rfc.CrossValScore(mean=True))
# print (clf)
#rfc.plot_learning_curve(cv=5,plt=True)