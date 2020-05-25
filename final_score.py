#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:40:31 2019

@author: lvguanxun
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from ipykernel.kernelapp import IPKernelApp
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



#21127 ~ 24094   2012~ 2016
df = pd.read_csv('Seasons_Stats.csv')
data = df[21127:24094]
test_data = df[24096:]

def run_PPG(data):
    PPG = []
    for i in range(len(data)):
        temp = float(data.iloc[i]["PTS"]) / float(data.iloc[i]["G"])
        PPG.append(temp)

    data = data.reset_index()
#add prob to dataframe for old player
    data["PPG"] = pd.Series(PPG)
    data = data.drop(columns = ["index", "Unnamed: 0","Year","Pos","Tm","blanl", "blank2"])
    data = data.dropna()
    
    return data

train_data = run_PPG(data)
y = np.array(train_data["PPG"])
X = np.array(train_data.drop(columns = ["Player","PPG","PTS"]))


test_data = run_PPG(test_data)
test_y = test_data["PPG"].tolist()
test_data = np.array(test_data.drop(columns = ["Player","PPG","PTS"]))

from sklearn import preprocessing
X_stad = preprocessing.scale(X)
test_data_stad = preprocessing.scale(test_data)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(100, activation='relu', input_dim = 45))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
model.summary()

model.fit(X_stad, y, batch_size=10,nb_epoch=200)

    

NN_result = model.predict(test_data_stad)
MSE_NN = mean_squared_error(NN_result, test_y)

plt.figure(1)
plt.figure(figsize=(20, 8))
predict, = plt.plot(NN_result, '+', color ='blue', alpha=0.7)
original, = plt.plot(test_y, 'ro', color ='red', alpha=0.7)
plt.legend([predict, original], ["Predict", "Real values"])
plt.title('Prediction vs Real values in Nerual Network')
plt.show()
print("NN MSE: ", MSE_NN)



from sklearn.linear_model import LinearRegression
mod  =  LinearRegression()
mod.fit(X_stad, y)

LR_result = mod.predict(test_data_stad)
coef = mod.coef_
MSE_LR = mean_squared_error(LR_result, test_y)

plt.figure(2)
plt.figure(figsize=(20, 8))
predict, = plt.plot(LR_result, '+', color ='blue', alpha=0.7)
original, = plt.plot(test_y, 'ro', color ='red', alpha=0.7)
plt.legend([predict, original], ["Predict", "Real values"])
plt.title('Prediction vs Real values in LinearRegression')
plt.show()
print("LinearRegression MSE: ", MSE_LR)
