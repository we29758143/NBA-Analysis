A#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:42:01 2019

@author: lvguanxun
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Seasons_Stats.csv')

#player from 1980~2000, some of the starter stats is nan
o = data[5727:14966]
old_player_data = o.dropna(subset=["G", "GS"]) #drop nan in G/ GS

#player from 2013~2017
n = data[21679:]
new_player_data = n.dropna(subset=["G", "GS"])

#Calculate starter prob
starter_prob = []
for i in range(len(old_player_data)):
    temp = int(old_player_data.iloc[i]["GS"]) / int(old_player_data.iloc[i]["G"])
    starter_prob.append(temp)
    
starter_prob_new = []
for i in range(len(new_player_data)):
    temp = int(new_player_data.iloc[i]["GS"]) / int(new_player_data.iloc[i]["G"])
    starter_prob_new.append(temp)

old_player_data = old_player_data.reset_index()
#add prob to dataframe for old player
old_player_data["star_prob"] = pd.Series(starter_prob)
old_player_data["all_star_label"] = pd.Series(starter_prob)

new_player_data = new_player_data.reset_index()
#add prob to dataframe for new player
new_player_data["star_prob"] = pd.Series(starter_prob_new)
new_player_data["all_star_label"] = pd.Series(starter_prob_new)

# df["all_star_label"][i] =1
#filter out all star
def all_star(df):
    j = 0
    for i in range(len(df)):
        if (float(df.iloc[i]["PER"]) > 15) and (float(df.iloc[i]["star_prob"]) > 0.8):
            df.loc[i,"all_star_label"] = 1
            j+=1
        else:
            df.loc[i,"all_star_label"] = 0
  
    return df,j

#fill 0 to nan 3%
old_player_data, num_old_star = all_star(old_player_data)
old_player_data["3P%"] = old_player_data["3P%"].fillna(0)
old_player_data["FT%"] = old_player_data["FT%"].fillna(0)

new_player_data, num_new_star = all_star(new_player_data)
new_player_data["3P%"] = new_player_data["3P%"].fillna(0)
new_player_data["FT%"] = new_player_data["FT%"].fillna(0)

#drop two nan columns
old_player_data = old_player_data.drop(columns = ["blanl", "blank2"])
new_player_data = new_player_data.drop(columns = ["blanl", "blank2"])

#create train and test data
train_old_data = old_player_data.drop(columns = ["index", "Unnamed: 0","Year", "Player","Pos","Tm"
                                                 ,"star_prob"])
train_old_data = train_old_data.dropna()

test_new_data = new_player_data.drop(columns = ["index","Unnamed: 0","Year", "Player","Pos","Tm"
                                                ,"star_prob"])
test_new_data = test_new_data.dropna()

label_old_star = train_old_data["all_star_label"]
label_new_star = test_new_data["all_star_label"]

#drop nan player
train_old_data = train_old_data.dropna()


#normalize
from sklearn import preprocessing
normalized_X = preprocessing.normalize(train_old_data,axis=0)
normalized_X_drop_label = normalized_X[:,:-1]

#=============split data into validation data and training data============
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(normalized_X_drop_label, label_old_star, test_size=0.1, random_state=42)

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3), 
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(), #have weight
    SVC(kernel="linear", C=0.025, probability=True), #have weight
    DecisionTreeClassifier()
    ]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
print("================Validation==============================")
print("================Validation==============================")
print("================Validation==============================")
print("================Validation==============================")
for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Number of All-star: " ,sum(train_predictions))
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions_prob = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions_prob)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)

#LinearDiscriminantAnalysis, SVC have weight
#    weight = clf.coef_
#    print('weights: ', clf.coef_)
    
print("="*30)
import seaborn as sns
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()
print("================Validation==============================")
print("================Validation==============================")
print("================Validation==============================")
print("================Validation==============================")

#=================input new_test data into model=====================
#=================input new_test data into model=====================
#=================input new_test data into model=====================
#=================input new_test data into model=====================

normalized_X_new = preprocessing.normalize(test_new_data,axis=0)
test_data = normalized_X_new[:,:-1] #final test data



# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log_n = pd.DataFrame(columns=log_cols)
print("================Test set==============================")
print("================Test set==============================")
print("================Test set==============================")
print("================Test set==============================")
for clf in classifiers:
    clf.fit(X_train, y_train)
    name_n = clf.__class__.__name__
    
    print("="*30)
    print(name_n)
    
    print('****Results****')
    train_predictions_n = clf.predict(test_data)
    acc_n = accuracy_score(label_new_star, train_predictions_n)
    print("Accuracy: {:.4%}".format(acc_n))
    print("Number of All-star: " ,sum(train_predictions_n))
    
    train_predictions_n_prob = clf.predict_proba(test_data)
    ll_n = log_loss(label_new_star, train_predictions_n_prob)
    print("Log Loss: {}".format(ll_n))
    
    log_entry_n = pd.DataFrame([[name_n, acc_n*100, ll_n]], columns=log_cols)
    log_n = log.append(log_entry_n)
    
    
print("="*30)
import seaborn as sns
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log_n, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log_n, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()
print("================Test set==============================")
print("================Test set==============================")
print("================Test set==============================")
print("================Test set==============================")



