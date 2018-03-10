from Feature_Generation import result,  y_gen,test
from sklearn.feature_selection import RFE
import lightgbm as lgb
from model_gbdt import clf_gbc
from model_lightGMB import clf_lgbc
from model_rf import clf_rf
from model_xgboost import clf_xgbc
from model_extratree import clf_extc
from sklearn.svm import SVC
from modeling import Stack_Ensemble


def Select_features(n_features):
    lgbc_params = {'learning_rate':0.1,
                    'max_depth': 6,
                    'min_child_weight': 4,
                    'n_estimators':100,
                    'subsample':0.95,
                    'colsample_bytree':0.6,
                    'seed':7,
                    'num_leaves':65,
                    'verbose':-1}
    estimator = lgb.LGBMClassifier(**lgbc_params)
    selector = RFE(estimator, n_features, step=3)
    selector = selector.fit(result, y_gen)
    cols_rfe = result.columns[selector.get_support()]
    new_result  = result[cols_rfe]


    columns_select = [i for i in new_result.columns]
    new_test = test[columns_select]

    return new_test,new_result

dict_result = {}

for n in range(10,29):
    new_test, new_result = Select_features(n)
    base_learners = [clf_lgbc, clf_gbc, clf_rf, clf_extc, clf_xgbc]
    stacker_parameters = {"kernel": ['rbf'],
                          "gamma": [0.001, 0.01, 0.1],
                          'C': [200, 300, 1000]}
    stacker_estimator = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
                            decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
                            max_iter=-1, probability=True, random_state=None, shrinking=True,
                            tol=0.001, verbose=False)
    ensemble = Stack_Ensemble(n_splits=5, stacker=stacker_estimator, GridParams=stacker_parameters, Grid=True,
                              base_models=base_learners, cv=5, scoring='accuracy')
    score = ensemble.fit_predict(new_result, y_gen, new_test, get_result=False)
    dict_result[n] = score
print  (dict_result)
