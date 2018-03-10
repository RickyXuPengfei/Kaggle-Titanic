#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:56:00 2018

@author: ricky_xu
"""

from modeling import Classifier
from Feature_Generation import y_gen
from Feature_selection import new_result
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

lgbc_parameters={'n_estimators':[i*100 for i in range(1,10)]
                }

lgbc_params = {'learning_rate':0.1,
              'max_depth': 6,
              'min_child_weight': 4,
              'n_estimators':100,
              'subsample':0.95,
              'colsample_bytree':0.6,
              'seed':7,
              'num_leaves':65,
              'verbose':-1}

lgbc =Classifier(lgb.LGBMClassifier,"lgb",new_result,y_gen,seed=0,params=lgbc_params)
# lgbc.train()
print ("LightBoost Training")
#lgbc.GridSearch(lgbc_parameters)
clf_lgbc=lgbc.clf
#print (lgbc.feature_importance())
#sns.barplot(x='Features',y='Feature_Importances',data=lgbc.feature_importance())
#plt.show()
# print (lgbc.CrossValScore(mean=True))