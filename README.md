
Kaggle-Titanic
==============
![] (images/logo.png)

Description
-----------

Analysis of what sorts of people were likely to survive in the titanic tragedy. Apply the tools of machine learning to predict which passengers survived the tragedy. I build an ensemble classifier and perform all the analyses , feature engineering ... and achieved accuracy score of **81.3% (0.81339)** in Public Leaderboard. And the rank of my solution is **Top 5%** (433rd/10262). 


Dependencies
-----------
* Numpy
* Pandas 
* scikit-learn
* matplotlib
* seaborn

FlowChart
---------
![] (images/FlowChart.png)

Project Files
-------------
### data
- train.csv :  training data.
- test.csv : test data.
- train_gen.csv : training data after feature_Handling. 
- test_gen.csv : test data after feature_Handling.
- dataset_gen.csv : data combining training data and test data after feature_Handling.

### src
- Feature_Handling.py :  visulize and process the features of training data and test data.
- Feature_Generation.py : generate new features from data feature_Handling.
- Feature_selection.py : select the best features.
- N_features_tunning.py : determine the number of features to be selected.
- modeling.py : encapsulate some commone functions in sklearn in order to train a model; realize stacking ensemble method.
- model_extratree.py :  ExtraTreesClassifier fits the data.
- model_gbdt.py : GradientBoostingClassifier fits the data.
- model_lightGMB.py : LGBMClassifier fits the data.
- model_rf.py : RandomForestClassifier fits the data.
- model_xgboost.py : XGBClassifier fits the data.

### submission
- summit.csv : the prediction of my solution.