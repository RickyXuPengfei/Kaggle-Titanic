#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:13:15 2018

@author: ricky_xu
"""


from modeling import Classifier
import xgboost as xgb
from Feature_Generation import X_gen,y_gen

xgb_params = {'learning_rate':0.1,
              'max_depth': 4,
              'min_child_weight': 3,
              'n_estimators':100,
              'gamma':0.35,
              'subsample':0.95,
              'colsample_bytree':0.6,
              'scale_pos_weight':1,
              'seed':1337,
              'base_score':0.5,
              'colsample_bylevel' :1,
              'reg_alpha': 0
                }

xgb_parameters = {'seed':range(10)}

xgbc = Classifier(xgb.XGBClassifier, 'xgbc', X_gen, y_gen, seed=0, params=xgb_params, scoring='accuracy')
# xgbc.train()
print("XGBoost training now!")
#xgbc.GridSearch(xgb_parameters)
clf_xgbc=xgbc.clf
#print (xgbc.feature_importance())
# print(xgbc.CrossValScore(mean=True))
#xgbc.plot_feature_importance(True)
#xgb_clf=xgbc.clf
#print (xgb_clf)
#xgb.plot_importance(xgbc.clf)
xgbc.plot_learning_curve(cv=5,plt=True)
