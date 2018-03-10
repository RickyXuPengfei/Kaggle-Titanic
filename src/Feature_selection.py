#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:57:41 2018

@author: ricky_xu
"""

from Feature_Generation import result, X_gen, y_gen,test
from sklearn.feature_selection import RFE
import lightgbm as lgb


lgbc_params = {'learning_rate':0.1,
              'max_depth': 6,
              'min_child_weight': 4,
              'n_estimators':100,
              'subsample':0.95,
              'colsample_bytree':0.6,
              'seed':7,
              'num_leaves':65,
              'verbose':-1}
# print (result.shape[1])
estimator = lgb.LGBMClassifier(**lgbc_params)
selector = RFE(estimator, 25, step=3)
selector = selector.fit(result, y_gen)
cols_rfe = result.columns[selector.get_support()]
new_result  = result[cols_rfe]


columns_select = [i for i in new_result.columns] 
new_test = test[columns_select]

