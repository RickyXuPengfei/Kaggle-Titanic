#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:56:00 2018

@author: ricky_xu
"""

from modeling import Classifier
from sklearn.ensemble import GradientBoostingClassifier
from Feature_Generation import y_gen
from Feature_selection import new_result

gb_parameters={ 
               'subsample':[0.75,0.8,0.85]}

gb_params = {'learning_rate': 0.1, 
             'loss': 'deviance', 
             'max_depth': 5, 
             'max_features': 0.5, 
             'min_samples_split': 200,
             'min_samples_leaf': 40, 
             'n_estimators': 200,
             'random_state':2}
gbc =Classifier(GradientBoostingClassifier,'gbc',new_result,y_gen,seed=0,params=gb_params)
# gbc.train()
print ("GbBC Training")
#gbc.GridSearch(gb_parameters)
clf_gbc=gbc.clf
#print (gbc.feature_importance())
#sns.barplot(x='Features',y='Feature_Importances',data=gbc.feature_importance())
#plt.show()
# print (gbc.CrossValScore(mean=True))
#gbc.plot_learning_curve(cv=5,plt=True)