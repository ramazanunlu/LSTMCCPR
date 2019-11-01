#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:16:10 2019

@author: ramazanunlu
"""

import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,GRU,SimpleRNN
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.initializers import Initializer
from keras import optimizers
from keras import regularizers
from sklearn import metrics
from keras.layers import Bidirectional
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import datamaker as makedata
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


#%%
w=30
alpha=np.arange(1.355,1.806,0.025) #np.arange(0.005,1.805,0.025)
#alpha=[0.45]
abtype=5
normalsize=950
abnormalsize=50
index=pd.DataFrame()
for iter in alpha:
    df=makedata.GenerateData(w,iter,abtype,normalsize,abnormalsize)
    X=df[:,:-1]
    y=df[:,-1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    SVM_Model= svm.SVC(kernel='poly',C=1,degree=3)
    SVM_Model.fit(X_train, y_train.reshape(800,))

    WSVM_Model = svm.SVC(kernel='poly',class_weight={1:0.526316,0:10},degree=2)
    WSVM_Model.fit(X_train, y_train.reshape(800,))

    w2=100
    data=makedata.GenerateData(w2,iter,abtype,1,1)
    #plt.plot(data[0,:])
    #plt.plot(data[1,:])

    data=np.delete(data,-1,1)
    data=data.reshape(2*w2,1)
    series=pd.DataFrame(data)
    series_s = series.copy()

    for i in range(29):
            series = pd.concat([series, series_s.shift(-(i+1))], axis = 1)
            series.dropna(axis=0, inplace=True)

    y_pred_SVM=SVM_Model.predict(series)
    y_pred_WSVM=WSVM_Model.predict(series)

    #SVM_compareindex= ((indexOfFirstOne(y_pred_SVM, len(y_pred_SVM))+(w-1))-99)/w
    
    
    
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
        
    y_train= to_categorical(y_train)
    y_test= to_categorical(y_test)
    batch_size=50

    model = Sequential()
    #model.add(Bidirectional(LSTM(16,activation='tanh',batch_input_shape=(batch_size,15,1),return_sequences=False,kernel_initializer='RandomNormal',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01),stateful=True)))
    model.add(Bidirectional(LSTM(32,activation='tanh',input_shape=(X_train.shape[1:]),return_sequences=False)))
    #model.add(Bidirectional(LSTM(2,activation='tanh',input_shape=(X_train.shape[1:]),return_sequences=False)))
    #model.add(Dense(4,activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    opt=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-3, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True)
    #print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test),callbacks=[monitor,checkpointer],epochs=1000, batch_size=batch_size,verbose=0)
    model.load_weights('best_weights.hdf5')

    series=np.array(series)
    series = series.reshape(series.shape[0],series.shape[1],1)

    y_pred_LSTM = model.predict(series,batch_size=batch_size)
    y_pred_LSTM = np.argmax(y_pred_LSTM,axis=1)
    #LSTM_compareindex= ((indexOfFirstOne(y_pred_LSTM, len(y_pred_LSTM))+(w-1))-99)/w
    
    
    
    
    
   
    if np.sum(y_pred_SVM == 0) >= 3:
        idx=np.array(np.where(y_pred_SVM == 0))[0,0:3]
    else:
        A=np.array(np.where(y_pred_SVM == 0))
        idx=A[0,0:len(A)]
        
    SVM_compareindex=((idx+(w-1))-99)/w
    
    for j in range(len(idx)):
        if SVM_compareindex.size == 0 :
            SVM_compareindex=np.zeros((len(idx),))
            break
        elif SVM_compareindex[j] < 0:
            SVM_compareindex[j]=0
        elif SVM_compareindex[j] > 1:
            SVM_compareindex[j]=-1
        else:
            SVM_compareindex[j]=SVM_compareindex[j]
    if len(SVM_compareindex) < 3:
        SVM_compareindex=np.append(SVM_compareindex, np.repeat(np.nan, 3-len(SVM_compareindex)))
    #WSVM_compareindex= ((indexOfFirstOne(y_pred_WSVM, len(y_pred_WSVM))+(w-1))-99)/w
    if np.sum(y_pred_WSVM == 0) >= 3:
        idx=np.array(np.where(y_pred_WSVM == 0))[0,0:3]
    else:
        A=np.array(np.where(y_pred_WSVM == 0))
        idx=A[0,0:len(A)]
    WSVM_compareindex=((idx+(w-1))-99)/w
    
    for j in range(len(idx)):
        if WSVM_compareindex.size == 0 :
            WSVM_compareindex=np.zeros((len(idx),))
            break
        elif WSVM_compareindex[j] < 0 :
            WSVM_compareindex[j]=0
        elif WSVM_compareindex[j] > 1:
            WSVM_compareindex[j]=-1
        else:
            WSVM_compareindex[j]=WSVM_compareindex[j]
    if len(WSVM_compareindex) < 3:
        WSVM_compareindex=np.append(WSVM_compareindex, np.repeat(np.nan, 3-len(WSVM_compareindex)))        
            
    if np.sum(y_pred_LSTM == 0) >= 3:
        idx=np.array(np.where(y_pred_LSTM == 0))[0,0:3]
    else:
        A=np.array(np.where(y_pred_LSTM == 0))
        idx=A[0,0:len(A)]
    
    LSTM_compareindex=((idx+(w-1))-99)/w
    for j in range(len(idx)):
         if LSTM_compareindex.size == 0 :
            LSTM_compareindex=np.zeros((len(idx),))
            break
         elif LSTM_compareindex[j] < 0 or  LSTM_compareindex.size == 0:
             LSTM_compareindex[j]=0
         elif LSTM_compareindex[j] > 1:
             LSTM_compareindex[j]=-1
         else:
             LSTM_compareindex[j]=LSTM_compareindex[j] #
    if len(LSTM_compareindex) < 3:
        LSTM_compareindex=np.append(LSTM_compareindex, np.repeat(np.nan, 3-len(LSTM_compareindex)))    
    a=np.insert(np.concatenate((SVM_compareindex,WSVM_compareindex,LSTM_compareindex)),0,iter)
    indices=a.reshape(1,a.shape[0])
    index=pd.concat([index , pd.DataFrame(indices)])

index.to_csv("comparisonresults.csv")
