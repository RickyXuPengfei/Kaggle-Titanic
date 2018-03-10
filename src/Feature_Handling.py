#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:51:38 2018

@author: ricky_xu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.preprocessing import LabelEncoder 


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


#Constant row(each feature is same)
constant = train.nunique(axis=1)==1
#print (constant[constant==True].index)

#duplicated 
#without id
duplicated_train = train.iloc[:,lambda df:np.arange(1,train.shape[1])].duplicated()
#print (duplicated_train[duplicated_train==True])



#Outilers
deal_columns = [index for index in train.dtypes[train.dtypes!=object].index]
for column in ['PassengerId', 'Survived', 'Pclass']:
    deal_columns.remove(column)
#print (deal_columns)


def outiliers(df,n_feature, deal_columns):
    """
    reveice the columns belongs to outiliers.

    :param df: DataFrame; the data to be dealt with

    :param n_feature: int; the number of features to deal

    :param deal_columns: list; a list of columns to be deal

    :return: list; a list of columns which are outiliers
    """
    outlier_indices = []
    
    for col in deal_columns:
        col_value = df[col].value_counts().sort_values()
        col_bound = np.percentile(col_value,1)
        col_index = col_value[col_value==col_bound].index
        outlier_list_col = df[df[col].isin(col_index)].index
        outlier_indices.extend(outlier_list_col)
        print (col)
        print (col_index)
        print (outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = [k for k,v in outlier_indices.items() if v>=n_feature]
    
    return multiple_outliers
#print (outiliers(train,2, deal_columns))

def detect_outiliers(df,n, features):
    """
    reveice the columns belongs to outiliers.

    :param df: DataFrame; the data to be dealt with.

    :param n: int; the threshold number of the outlied columns.

    :param features: list; a list of columns to be deal.

    :return:list; a list of columns which are outiliers.
    """
    outlier_indices = []
    for col in features:
        col_value = df[col].value_counts(normalize=True).index
        Q1 = np.percentile(col_value, 25)
        Q3 = np.percentile(col_value, 75)
        IQR  = Q3 - Q1
        
        outlier_step = 1.5*IQR
        
        outlier_list_col = df[(df[col]<Q1-outlier_step)|(df[col]>Q3+outlier_step)].index
        
    outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = [k for k, v in outlier_indices.items() if v>=n]
    return multiple_outliers

def set_index(train,test):
    """
    combine the train data and test data, then merge the index.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :return: tuple; a tuple of train,test data after processing.
    """
    list_iter=[train,test]
    list_return=[]
    for i in list_iter:
        i = i.set_index(i['PassengerId']-1)
        i.index.name = None
        list_return.append(i)
    return tuple(list_return)
train,test = set_index(train,test)
dataset = pd.concat([train,test],axis=0)

#Missing value
#print (train.isnull().sum())
#print (test.isnull().sum())
#print (train['SibSp'].value_counts())
def plot_Age(plot=False):
    """
    plot the subplots of Age feature.

    :param plot: boolean; determine whether it can be plotted.
    """
    if plot==True:
        fig,axes = plt.subplots(nrows=2,ncols=2)
        age = dataset.loc[~dataset['Age'].isnull(),'Age']
        a1 = sns.distplot(age,bins=20, kde=True,rug=True,ax=axes[0,0])
        a1.axes.set_title("Age of dataset")

        age_train = train.loc[~train['Age'].isnull(),['Age','Survived']]
        age_test = test.loc[~test['Age'].isnull(),'Age']
        a2 = sns.distplot(age_test,bins=20,kde=True, rug=True, ax=axes[0,1])
        a2.axes.set_title("Age of test")

        a3 = sns.distplot(age_train['Age'],bins=20,kde=True, rug=True, ax=axes[1,0])
        a3.axes.set_title("Age of train")

        a4 = sns.distplot(age_train.loc[age_train['Survived']==1,'Age'],bins=20,kde=True, rug=True, ax=axes[1,1])
        a4.axes.set_title("Age of train By Survived")
        plt.tight_layout()
        plt.show()

def plot_AgeWithSurvived():
    """
    plot the kernel density estimate of Age grouped by Survived.

    """
    facet = sns.FacetGrid(train, hue="Survived", aspect=8)
    facet.map(sns.kdeplot, 'Age', shade=True)
    facet.set(xlim=(0, train['Age'].max()))
    facet.add_legend()
    plt.show()


def Age_feature(train, test, dataset, *cols):
    """
    group the columns which is related with Age. use the mean of Age in each group to fill the nan in Age feature.

    :param train: DataFrame; train data.

    :param test:  DataFrame; test data.

    :param dataset: DataFrame; dataset data merging train and test.

    :param cols: list; a list of columns related with Age feature.

    :return: tuple ; a tuple containing train, test, dataset.
    """
    data = dataset.groupby(cols)['Age'].transform(lambda x: x.fillna(x.mean()))
    list_iter = [train,test,dataset]
    list_return = []
    for i in list_iter:
        i['Age_IsNull'] = i['Age'].map(lambda x:1 if np.isnan(x) else 0)
        i['Age'] =  (data.iloc[list(i.index)])
        list_return.append(i)
    return tuple(list_return)

#train, test, dataset = Age_feature(train, test, dataset, 'Sex')

#Fare
# use the mean of Fare feature to fill the nan of Fare feature.
test['Fare'].fillna(dataset['Fare'].mean(), inplace = True)
dataset['Fare'].fillna(dataset['Fare'].mean(), inplace = True)

def plot_Fare(plot=False):
    """
    plot the subplots of Fare feature.

    :param plot: boolean; determine whether it can be plotted.

    """

    if plot==True:
        fig,axis=plt.subplots(nrows=2,ncols=2)
        a1 = sns.distplot(dataset['Fare'],bins=20,rug=True,ax=axis[0,0])
        
        a1.axes.set_title("Fare of dataset")
    
        a2 = sns.distplot(train['Fare'],bins=20,rug=True,ax=axis[0,1])
        a2.axes.set_title("Fare of Train")
    
        a3 = sns.boxplot(y='Fare',data=train,ax=axis[1,0])
        a3.axes.set_title("Fare of Train Of Box")
        a3.axes.set_ylabel("Fare")
    
#    a4 = sns.FacetGrid(train, hue="Survived", ax=axis[1,1])
#    a4.map(sns.kdeplot, 'Fare', shade=True)
#    a4.set(xlim=(0, train['Fare'].max()))
#    a4.add_legend()
        a4 = sns.kdeplot(train['Fare'],ax=axis[1,1])
        a4.set(xlim=(0, train['Fare'].max()))
        a4.axes.set_title('FareCount')

        plt.tight_layout()
        plt.show()
#plot_Fare(plot=True)

def plot_FareWithSurvived(plot=False):
    """
    plot the kernel density estimate of Age grouped by Survived.

    :param plot:  boolean; determine whether it can be plotted.

    """

    if plot==True:
        facet = sns.FacetGrid(train, hue="Survived", aspect=8)
        facet.map(sns.kdeplot, 'Fare', shade=True)
        facet.set(xlim=(0, train['Fare'].max()))
        facet.add_legend()
        plt.show()
#plot_FareWithSurvived(plot=True)


def Fare_group(train,test,dataset):
    """

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """
    list_iter=[train,test,dataset]
    list_return=[]
    for i in list_iter:
        i['Fare_Group'] = np.where(i['Fare']<17.18,0,1)
        list_return.append(i)
    return tuple(list_return)

#Pclass
def plot_Pclass(plot=False):
    """
    plot the subplots of Pclass feature.

    :param plot: boolean; determine whether it can be plotted.
    """
    if plot==True:
        fig,axis=plt.subplots(nrows=2,ncols=2)
        a1 = sns.violinplot(x='Pclass',y='Survived',data=train,ax=axis[0,0])
        a1.axes.set_title("Plcass of Train")
    
        a2 = sns.barplot(x='Pclass',y='Survived',data=train,ax=axis[0,1])
        a2.axes.set_title("Fare of Train")
    
        a3 = sns.boxplot(x='Pclass',y='Survived',data=train,ax=axis[1,0])
        a3.axes.set_title("Pclass of Train BoxPlot ")
#        a3.axes.set_ylabel("Fare")

        a4 = sns.countplot('Pclass',data=train,hue='Survived',ax=axis[1,1])
        a4.axes.set_title('PclassCount')

        plt.tight_layout()
        plt.show()
##plot_Pclass(plot=True)
def Pclass_Frequency(train,test,dataset):
    """

    generate the frequency of Pclass feature.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """
    deal_list=[train,test,dataset]
    return_list = []
    for i in deal_list:
        i['Pclass_Frequency'] = i['Pclass'].map(i.groupby(['Pclass']).size()/i.shape[0])
        return_list.append(i)
    return tuple(return_list)

#train,test,dataset = Pclass_Frequency(train,test,dataset)

#Sex
#print (dataset['Sex'].value_counts())

def plot_Sex(plot=False):

    """
    plot the subplots of Sex Feature.

    :param plot: boolean; determine whether it can be plotted.
    """

    if plot==True:
        fig,axis=plt.subplots(nrows=2,ncols=2)
        a1 = sns.countplot(x='Sex',data=dataset,ax=axis[0,0])
        a1.axes.set_title("SexCount of dataset")
        
        a2 = sns.countplot(x='Sex',data=train,ax=axis[0,1])
        a2.axes.set_title("SexCount of Train")
    
        a3 = sns.barplot(x='Sex',y='Survived',data=train,ax=axis[1,0])
        a3.axes.set_title("SexSurvival of Train ")
        a3.axes.set_ylabel("Survival Rate")
       

        a4 = sns.countplot(x='Sex',data= train, hue = 'Survived',ax=axis[1,1])
        a4.axes.set_title("SurivalCount of Train By Sex ")
        a4.axes.set_ylabel("Survival Count")
        _,legend_label = a4.axes.get_legend_handles_labels()
        new_legend = [legend_mapping[i] for i in legend_label]
        a4.axes.legend(new_legend)

        plt.tight_layout()
        plt.show()
#plot_Sex(plot=True)

#print (pd.factorize(train['Sex']))
legend_mapping = {'0':'NonSurvived','1':'Survived'}

def Sex_Frquency(train,test,dataset):
    """
    generate the frequency of Sex.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """
    deal_list = [train,test,dataset]
    return_list = []
    map_frequency = deal_list[2].groupby(['Sex']).size()/deal_list[2].shape[0]
    for i in deal_list:
        i['Sex_Frequency'] = i['Sex'].map(map_frequency)
        return_list.append(i)
    return tuple(return_list)

def Sex_Label(train,test,dataset):
    """
    labelencode the Sex feature.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """
    deal_list =[train,test,dataset]
    return_list= []
    le = LabelEncoder()
    le.fit(deal_list[2]['Sex'])
    dict_sex = {value:index for index, value in enumerate(le.classes_)}
    for i in deal_list:
        i['Sex_label'] = i['Sex'].map(dict_sex)
        return_list.append(i)
    return_list.append(dict_sex)
    return tuple(return_list)

#train,test,dataset = Sex_Frquency(train,test,dataset)
#train,test,dataset,Sex_Mapping = Sex_Label(train,test,dataset)

#Embarked
#print (dataset['Embarked'].isnull().sum())
max_Emarked =dataset['Embarked'].value_counts().index[0]

for i in [train,test,dataset]:
    i['Embarked'] = i['Embarked'].fillna(max_Emarked)

def plot_Embarked(plot=False):
    """
    plot he subplots of Embarked feature.

    :param plot: boolean; determine whether it can be plotted.
    """

    if plot==True:
        fig,axis=plt.subplots(nrows=2,ncols=2)
        a1 = sns.countplot(x='Embarked',data=dataset,ax=axis[0,0])
        a1.axes.set_title("EmbarkedCount of dataset")
        
        a2 = sns.countplot(x='Embarked',data=train,ax=axis[0,1])
        a2.axes.set_title("EmbarkedCount of Train")
    
        a3 = sns.barplot(x='Embarked',y='Survived',data=train,ax=axis[1,0])
        a3.axes.set_title("EmbarkedSurvival of Train ")
        a3.axes.set_ylabel("Survival Rate")
       

        a4 = sns.countplot(x='Embarked',data= train, hue = 'Survived',ax=axis[1,1])
        a4.axes.set_title("SurivalCount of Train By Embarked ")
        a4.axes.set_ylabel("Survival Count")
        _,legend_label = a4.axes.get_legend_handles_labels()
        new_legend = [legend_mapping[i] for i in legend_label]
        a4.axes.legend(new_legend)

        plt.tight_layout()
        plt.show()
        

def Embarked_Frquency(train,test,dataset):
    """
    generate the frequency of Sex.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """
    deal_list = [train,test,dataset]
    return_list = []
    map_frequency = deal_list[2].groupby(['Embarked']).size()/deal_list[2].shape[0]
    for i in deal_list:
        i['Embarked_Frequency'] = i['Embarked'].map(map_frequency)
        return_list.append(i)
    return tuple(return_list)

def Embarked_Label(train,test,dataset):
    """
    labelencode Embarked feature.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """
    deal_list =[train,test,dataset]
    return_list= []
    le = LabelEncoder()
    le.fit(deal_list[2]['Embarked'])
    dict_Embarked = {value:index for index, value in enumerate(le.classes_)}
    for i in deal_list:
        i['Embarked_label'] = i['Embarked'].map(dict_Embarked)
        return_list.append(i)
    return_list.append(dict_Embarked)
    return tuple(return_list)
#
#train,test,dataset = Embarked_Frquency(train,test,dataset)
#train,test,dataset,Embarked_Mapping= Embarked_Label(train,test,dataset)

#SibSp #Parchcc
#print (np.corrcoef(dataset['SibSp'],dataset['Parch']))
#for col in ['SibSp','Parch']:
#    print (dataset[col].isnull().sum())

##Generate new feature "Family_size" (Family_size = SibSp + Parch)
for i in [train,test,dataset]:
    i['Family_size'] = i['SibSp']+i['Parch']
    del i['SibSp']
    del i['Parch']
    
def plot_Family_size(plot=False):
    """
    plot the Family_size

    :param plot: boolean; determine whether it can be plotted.
    """

    if plot==True:
        fig,axis=plt.subplots(nrows=2,ncols=2)
        a1 = sns.countplot(x='Family_size',data=dataset,ax=axis[0,0])
        a1.axes.set_title("Family_sizeCount of dataset")
        
        a2 = sns.countplot(x='Family_size',data=train,ax=axis[0,1])
        a2.axes.set_title("Family_sizeCount of Train")
    
        a3 = sns.barplot(x='Family_size',y='Survived',data=train,ax=axis[1,0])
        a3.axes.set_title("Family_sizeSurvival of Train ")
        a3.axes.set_ylabel("Survival Rate")
       

        a4 = sns.countplot(x='Family_size',data= train, hue = 'Survived',ax=axis[1,1])
        a4.axes.set_title("SurivalCount of Train By Embarked ")
        a4.axes.set_ylabel("Survival Count")
        _,legend_label = a4.axes.get_legend_handles_labels()
        new_legend = [legend_mapping[i] for i in legend_label]
        a4.axes.legend(new_legend)

        plt.tight_layout()
        plt.show()
#plot_Family_size(plot=True)

def plot_FamilySizeWithSurvived(plot=False):
    """
    plot the Family_size grouped by Survived.

    :param plot: boolean; determine whether it can be plotted.
    """
    if plot==True:
        facet = sns.FacetGrid(train, hue="Survived", aspect=8)
        facet.map(sns.kdeplot, 'Family_size', shade=True)
        facet.set(xlim=(0, train['Family_size'].max()))
        facet.add_legend()
        plt.show()
#plot_FamilySizeWithSurvived(plot=True)

def Family_Feature(train,test,dataset):
    """
    group Family_Feature manually depend on the result of plot_FamilySizeWithSurvived.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """

    deal_list = [train,test,dataset]
    return_list = []
    for i in deal_list:
        i['Family'] = np.where(i['Family_size']<1,0,
                         np.where(i['Family_size']<4,1,2))
        del i['Family_size']
        return_list.append(i)
    return tuple(return_list)

#train,test,dataset = Family_Feature(train,test,dataset)

def Family_Frquency(train,test,dataset):
    """
    generate the frequency of Family

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """
    deal_list = [train,test,dataset]
    return_list = []
    map_frequency = deal_list[2].groupby(['Family']).size()/deal_list[2].shape[0]
    for i in deal_list:
        i['Family_Frequency'] = i['Family'].map(map_frequency)
        return_list.append(i)
    return tuple(return_list)
#train,test,dataset = Family_Frquency(train,test,dataset)

#print (train[['Family','Family_Frequencywh']])

# New Feature : Ticket_lett  (mapping the value_counts of Ticket)
Ticket_Mapping = dataset['Ticket'].value_counts()
train['Ticket_lett'] = train['Ticket'].map(Ticket_Mapping)
test['Ticket_lett'] = test['Ticket'].map(Ticket_Mapping)
dataset['Ticket_lett'] = dataset['Ticket'].map(Ticket_Mapping)

def plot_TicketLettWithSurvived(plot=False):
    """
    plot the TicketLett grouped by Survived.

    :param plot: boolean; determine whether it can be plotted.
    """
    if plot==True:
        facet = sns.FacetGrid(train, hue="Survived", aspect=8)
        facet.map(sns.kdeplot, 'Ticket_lett', shade=True)
        facet.set(xlim=(0, train['Ticket_lett'].max()))
        facet.add_legend()
        plt.show()
#plot_TicketLettWithSurvived(plot=True)

def TicketLett_Feature(train,test,dataset):
    """
    generate the frequency of TicketLett.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """
    deal_list = [train,test,dataset]
    return_list = []
    for i in deal_list:
        i['TicketLettGroup'] = np.where(i['Ticket_lett']<0.6,0,
                         np.where(i['Ticket_lett']<1.35,1,
                        np.where(i['Ticket_lett']<5.6,2,
                       np.where(i['Ticket_lett']<9,3,4))))
        del i['Ticket_lett']
        return_list.append(i)
    return tuple(return_list)

#train,test,dataset = TicketLett_Feature(train,test,dataset)

def TicketGroup_Frequency(train,test,dataset):
    """
    generate the frequency of TicketGroup.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """

    deal_list = [train,test,dataset]
    return_list = []
    map_frequency = deal_list[2].groupby(['TicketLettGroup']).size()/deal_list[2].shape[0]
    for i in deal_list:
        i['TicketLettGroup_Frequency'] = i['TicketLettGroup'].map(map_frequency)
        return_list.append(i)
    return tuple(return_list)

#train,test,dataset = TicketGroup_Frequency(train,test,dataset)

#name
def extract_name(train,test,dataset):
    """
    new Feature Name_lett by extracting from name.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """

    deal_list = [train,test,dataset]
    return_list = []
    for i in deal_list:
        i['Name_lett'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        return_list.append(i)
    return tuple(return_list)
train,test,dataset = extract_name(train,test,dataset)


#NameLett_mapping = pd.factorize(dataset['Name_lett'])[1]
#Dict_NameMapping = {value:index for index,value in enumerate(NameLett_mapping)}
#train['NameLett_Mapping'] = train['Name_lett'].map(Dict_NameMapping)

def plot_NameLettWithSurvived(plot=False):
    """
    plot the Name_lett grouped by Survived.

    :param plot: boolean; determine whether it can be plotted.
    """
    if plot==True:
        facet = sns.FacetGrid(train, hue="Survived", aspect=8)
        facet.map(sns.kdeplot, 'NameLett_Mapping', shade=True)
        facet.set(xlim=(0, train['NameLett_Mapping'].max()))
        facet.add_legend()
        plt.show()
#plot_NameLettWithSurvived(plot=True)
#sns.countplot(x='Name_lett',hue='Survived',data=train)

def Name_Title_Code(x):
    """
    encode the Name_lett.

    :param x: String; Name_letter

    :return: int; integer label of Name_lett
    """
    if x == 'Mr.':
        return 1
    if (x == 'Mrs.') or (x=='Ms.') or (x=='Lady.') or (x == 'Mlle.') or (x =='Mme'):
        return 2
    if x == 'Miss.':
        return 3
    if x == 'Rev.':
        return 4
    return 5

# transform Name_lett into Name_title by using Name_Title_Code.
for i in [train,test, dataset]:
    i['Name_title'] = i['Name_lett'].map(Name_Title_Code)
    
def Name_title_Frequency(train,test,dataset):
    """
    generate the frequency of Name_title.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """
    deal_list = [train,test,dataset]
    return_list = []
    map_frequency = deal_list[2].groupby(['Name_title']).size()/deal_list[2].shape[0]
    for i in deal_list:
        i['Name_title_Frequency'] = i['Name_title'].map(map_frequency)
        return_list.append(i)
    return tuple(return_list)

#train,test,dataset = Name_title_Frequency(train,test,dataset)

def plot_CabinLettWithSurvived(plot=False):
    """
    plot the Cabin_group grouped by Survived.

    :param plot: boolean; determine whether it can be plotted.
    """
    if plot==True:
        facet = sns.FacetGrid(train, hue="Survived", aspect=8)
        facet.map(sns.kdeplot, 'Cabin_group', shade=True)
        facet.set(xlim=(0, train['Cabin_group'].max()))
        facet.add_legend()
        plt.show()

#plot_CabinLettWithSurvived(plot=True)

def CabinFetaure(train,test,dataset):
    """
    generate new Feature(Cabin_IsNull; Cabin_group) from Cabin.

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return:  tuple ; a tuple containing train, test, dataset.
    """

    deal_list = [train,test,dataset]
    return_list = []
    for i in deal_list:
        i['Cabin_IsNull'] = i['Cabin'].map(lambda x: 1 if pd.isnull(x) else 0)
        i['Cabin_lett'] = i['Cabin'].apply(lambda x:x.split(" ")[-1][1:] if not pd.isnull(x) else 0)
        i['Cabin_lett1'] = i['Cabin_lett'].apply(lambda x:int(x) if x !='' or x ==0 else np.nan)

        i['Cabin_lett1'] = np.where(i['Cabin_lett1']==0,np.nan,i['Cabin_lett1'])
        Cabin_count = i['Cabin_lett1'].value_counts()
        i['Cabin_group'] = i['Cabin_lett1'].map(Cabin_count)
        i['Cabin_group'] = i['Cabin_group'].fillna(0)
        del i['Cabin_lett']
        del i['Cabin_lett1']
        return_list.append(i)
    return tuple(return_list)
#train,test,dataset = CabinFetaure(train,test,dataset)

# use the handle functions to process the data.
train,test,dataset = Pclass_Frequency(train,test,dataset)
train,test,dataset = Sex_Frquency(train,test,dataset)
train,test,dataset,Sex_Mapping = Sex_Label(train,test,dataset)
train,test,dataset = Fare_group(train,test,dataset)
train,test,dataset = Embarked_Frquency(train,test,dataset)
train,test,dataset,Embarked_Mapping= Embarked_Label(train,test,dataset)
train,test,dataset = Family_Feature(train,test,dataset)
train,test,dataset = Family_Frquency(train,test,dataset)
train,test,dataset = TicketLett_Feature(train,test,dataset)
train,test,dataset = TicketGroup_Frequency(train,test,dataset)
train,test,dataset = extract_name(train,test,dataset)
train,test,dataset = Name_title_Frequency(train,test,dataset)
train,test,dataset = CabinFetaure(train,test,dataset)

columns = train.dtypes[train.dtypes==object].index

for i in [train,test,dataset]:
    for col in columns:
        del i[col]

def plot_corr():
    """
    plot the correlation of features.
    """
    ax = sns.heatmap(train.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
    plt.setp(ax.get_xticklabels(),rotation=20)
    plt.setp(ax.get_yticklabels(),rotation=10)
    plt.show()
#plot_corr()

train, test, dataset = Age_feature(train, test, dataset, 'Pclass','Name_title')

#plot_AgeWithSurvived()
def Age_Group(train,test,dataset):
    """
    group the Age using plot_AgeWithSurvived().

    :param train: DataFrame; train data.

    :param test: DataFrame; test data.

    :param dataset: DataFrame; dataset data.

    :return: tuple ; a tuple containing train, test, dataset.
    """
    deal_list = [train,test,dataset]
    return_list = []
    for i in deal_list:
        i['Age_group'] = np.where(i['Age']<17.4,0,
                             np.where(i['Age']<33.8,1,
                                 np.where(i['Age']<43.2,2,
                                    np.where(i['Age']<59,3,4))))
        return_list.append(i)
    return tuple(return_list)

train, test, dataset = Age_Group(train, test, dataset)
for i in [train, test, dataset]:
    for column in ['Age','Cabin_group']:
        i[column] = i[column].map(lambda x : int(x))


def get_x_y(df):
    """
    split the train data into two parts (feature data, target data)
    :param df:
    :return:
    """
    X=df.drop(['Survived','PassengerId'],axis=1)
    y=df['Survived']
    return X,y


X_train,y_train=get_x_y(train)
del test['PassengerId']