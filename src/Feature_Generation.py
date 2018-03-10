#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 17:25:51 2018

@author: ricky_xu
"""

import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("../data/train_gen.csv")
test = pd.read_csv("../data/test_gen.csv")
dataset = pd.read_csv("../data/dataset_gen.csv")
original_cols = set(train.columns)
#print (original_cols)
def agg_feature(agg_col,*group_col):
    """
    group the data by group_col(list); aggregate the each group using agg_col;
    calculate the mean, max, min of the aggregation to generate new features.

    :param agg_col:string; a feature which has high feature importance string.

    :param group_col:list; a list of columns related with agg_col.

    :return: tuple ; a tuple containing train, test, dataset.
    """
    L_return  = []
    for data in [train,test,dataset]:
        gp = data.groupby([i for i in group_col],as_index=False).agg(
                {agg_col:{name:function for name, function in  zip([agg_col+'_Mean',agg_col+'_Max',agg_col+'_Min'],[np.mean,np.max,np.min])}})
        
        gp.columns = [i for i in group_col]+[agg_col+'_Mean',agg_col+'_Max',agg_col+'_Min']
        
        data = pd.merge(data,gp,how='left',on=group_col)
        
        L_return.append(data)
    return tuple(L_return)
train,test, dataset = agg_feature("Fare","Pclass","TicketLettGroup","Cabin_IsNull")

train,test, dataset = agg_feature("Age","Pclass","Name_title")

#Cabin_IsNull(10/13 3/13) & Pclass
#mul
for i in [train,test,dataset]:
    i["Cabin_IsNull_num"] = i["Cabin_IsNull"].map({1:10/13,0:3/13})
    i['Cabin_IsNull_Pclass_Mul'] = i["Cabin_IsNull_num"] * i["Pclass"]
    del i["Cabin_IsNull_num"]
    encoder = i['Cabin_IsNull'].map(lambda x : str(x))+"|"+i['Pclass'].map(lambda x: str(x))
    le = LabelEncoder()
    i["Cabin_IsNull_Pclass"] = le.fit_transform(encoder)

#Cabin_group Pclass 
for i in [train,test,dataset]:
    i["Cabin_group_Pclass_div"] = i["Cabin_group"]/i["Pclass"]
    encoder = i['Cabin_group'].map(lambda x : str(x))+"|"+i['Pclass'].map(lambda x: str(x))
    le = LabelEncoder()
    i["Cabin_group_Pclass"] = le.fit_transform(encoder)

#Sex_label Name_title

for i in [train,test,dataset]:
    i["Sex_label_num"] = i["Sex_label"].map({1:0.85,0:0.45})
    i['Sex_label_Name_title_div'] = i["Sex_label_num"] / i["Name_title"]
    del i["Sex_label_num"]
    encoder = i['Sex_label'].map(lambda x : str(x))+"|"+i['Name_title'].map(lambda x: str(x))
    le = LabelEncoder()
    i["Sex_label_Name_title"] = le.fit_transform(encoder)


def feature_join(col1,col2,operate):
    """
    do some operations (mul; dive) on two categorical features which are related.

    :param col1: string; a categorical feature.

    :param col2: string; a categorical feature.

    :param operate: string; operation string ('mul'; 'div')

    :return:
    """
    return_list = []
    operate_col = col1 + col2 + operate
    for i in [train,test,dataset]:
        
        if operate == "mul":
            i[operate_col] = (i[col1]+1) * (i[col2]+1)
            
        elif operate == "div":
            i[operate_col] = (i[col1]+1) / (i[col2]+1)
        else:
            i[operate_col] = (i[col1]+1) + (i[col2]+1)
            
        encoder = i[col1].map(lambda x : str(x))+"|"+i[col2].map(lambda x: str(x))
        le = LabelEncoder()
        new_col = col1 + col2 
        i[new_col] = le.fit_transform(encoder)
        return_list.append(i)
    return tuple(return_list)

train, test, dataset = feature_join("Family","TicketLettGroup","mul")

#Cabin_IsNull  Cabin_group
train, test, dataset = feature_join("Cabin_IsNull","Cabin_group","div")

result_cols = set(train.columns)
cols = list(result_cols - original_cols)
for f in cols:
    train[f] = train[f].factorize()[0]
result = train.T.drop_duplicates().T

def get_x_y(df):
    X=df.drop(['Survived','PassengerId'],axis=1)
    y=df['Survived']

#    X_test=data_test.drop(['PassengerId'],axis=1).copy()
    return X,y

del result["Survived"]
result = result.drop(['Unnamed: 0', 'PassengerId'],axis=1)
#result = result.drop_duplicates()

# X_gen 合并了所有特征
X_gen = result
y_gen = train["Survived"]
