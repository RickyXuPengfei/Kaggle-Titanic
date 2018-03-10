#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:20:44 2018

@author: ricky_xu
"""

from model_gbdt import clf_gbc
from model_lightGMB import clf_lgbc
from model_rf import clf_rf
from model_xgboost import clf_xgbc
from model_extratree import clf_extc
from sklearn.svm import SVC
from Feature_Generation import y_gen
from Feature_selection import new_result,new_test
import pandas as pd
from modeling import Stack_Ensemble


base_learners = [clf_lgbc,clf_gbc,clf_rf,clf_extc,clf_xgbc]
stacker_parameters={"kernel":[ 'rbf'],
                "gamma":[0.001,0.01,0.1],
                'C':[200,300,1000]}
stacker_estimator = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
              max_iter=-1, probability=True, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
ensemble = Stack_Ensemble(n_splits =5, stacker =stacker_estimator, GridParams = stacker_parameters, Grid=True, base_models= base_learners,cv=5, scoring='accuracy')
Y_pred = ensemble.fit_predict(new_result,y_gen,new_test,get_result = True)
title = "stacking_27"
print (Y_pred)
data_test=pd.read_csv('test.csv')
Submission=pd.DataFrame({'PassengerId':data_test['PassengerId'],'Survived':Y_pred})
Submission['Survived'] = Submission['Survived'].map(lambda x:int(x))
Submission.to_csv('%s.csv'%(title),index=False)