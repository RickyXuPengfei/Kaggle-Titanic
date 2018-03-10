#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:54:04 2018

@author: ricky_xu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:56:00 2018

@author: ricky_xu
"""

from modeling import Classifier
from sklearn.ensemble import ExtraTreesClassifier
from Feature_Generation import y_gen
from Feature_selection import new_result

ext_parameters={
               'n_estimators':[100,200,300,400],
               }


ext_params = {'criterion':'gini',
             'n_estimators':100,
             'max_features':'sqrt',
             'max_depth': 7,
             'min_samples_leaf': 2,
             'random_state':5,
             'n_jobs':-1}

extc =Classifier(ExtraTreesClassifier,'extc',new_result,y_gen,seed=0,params=ext_params,scoring='accuracy')

# extc.train()
#extc.GridSearch(ext_parameters)
clf_extc=extc.clf
#print (extc.feature_importance())
#sns.barplot(x='Features',y='Feature_Importances',data=extc.feature_importance())
#plt.show()
# print (extc.CrossValScore(mean=True))
# print (extc)
# extc.plot_learning_curve(cv=5,plt=True)