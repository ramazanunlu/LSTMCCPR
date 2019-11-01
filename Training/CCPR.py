#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:39:19 2019

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
Uptrend_SVM=pd.DataFrame()
Uptrend_Weighted=pd.DataFrame()
Uptrend_LSTM=pd.DataFrame()

w=np.arange(10,20,10)
alpha=np.arange(0.005,0.2,0.02)
abtype=5
normalsize=950
abnormalsize=50

for j in w:
    for i in alpha:
        df=makedata.GenerateData(j,i,abtype,normalsize,abnormalsize)
        X=df[:,:-1]
        y=df[:,-1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        clf = svm.SVC(kernel='poly',C=1,degree=3)
        clf.fit(X_train, y_train.reshape(800,))
        y_pred=clf.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy score: {}".format(score))
        cnf_matrix_SVM = confusion_matrix(y_test, y_pred)
        sensitivity_SVM = cnf_matrix_SVM[0,0]/(cnf_matrix_SVM[0,0]+cnf_matrix_SVM[0,1])
        specificity_SVM = cnf_matrix_SVM[1,1]/(cnf_matrix_SVM[1,0]+cnf_matrix_SVM[1,1])
        Gmean_SVM=np.sqrt(sensitivity_SVM*specificity_SVM)
        dict_SVM = {'Window Length':j,'Parameter' : i, 'Sensitivity' : sensitivity_SVM,'Specificity':specificity_SVM,'G-Mean':Gmean_SVM,'Accuracy':score}
        Uptrend_SVM=pd.concat([Uptrend_SVM , pd.DataFrame([dict_SVM])])


        
        clf = svm.SVC(kernel='poly',class_weight='balanced',C=1,degree=3)
        clf.fit(X_train, y_train.reshape(800,))
        y_pred=clf.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy score: {}".format(score))
        cnf_matrix_weighted= confusion_matrix(y_test, y_pred)
        sensitivity_Weighted = cnf_matrix_weighted[0,0]/(cnf_matrix_weighted[0,0]+cnf_matrix_weighted[0,1])
        specificity_Weighted = cnf_matrix_weighted[1,1]/(cnf_matrix_weighted[1,0]+cnf_matrix_weighted[1,1])
        Gmean_Weighted=np.sqrt(sensitivity_Weighted*specificity_Weighted)
        dict_Weighted = {'Window Length':j,'Parameter' : i, 'Sensitivity' : sensitivity_Weighted,'Specificity':specificity_Weighted,'G-Mean':Gmean_Weighted,'Accuracy':score}
        Uptrend_Weighted=pd.concat([Uptrend_Weighted , pd.DataFrame([dict_Weighted])])
    
        
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
        pred = model.predict(X_test,batch_size=batch_size)
        pred = np.argmax(pred,axis=1)
        y_compare = np.argmax(y_test,axis=1) 
        score = metrics.accuracy_score(y_compare, pred)
        print("Accuracy score: {}".format(score))
        cnf_matrix_LSTM= confusion_matrix(y_compare, pred)
        sensitivity_LSTM = cnf_matrix_LSTM[0,0]/(cnf_matrix_LSTM[0,0]+cnf_matrix_LSTM[0,1])
        #print('Sensitivity : ', sensitivity1 )
        specificity_LSTM = cnf_matrix_LSTM[1,1]/(cnf_matrix_LSTM[1,0]+cnf_matrix_LSTM[1,1])
        #print('Specificity : ', specificity1)
        Gmean_LSTM=np.sqrt(sensitivity_LSTM*specificity_LSTM)
        dict_LSTM = {'Window Length':j,'Parameter' : i, 'Sensitivity' : sensitivity_LSTM,'Specificity':specificity_LSTM,'G-Mean':Gmean_LSTM,'Accuracy':score}

        Uptrend_LSTM=pd.concat([Uptrend_LSTM , pd.DataFrame([dict_LSTM])])

result = pd.concat([Uptrend_SVM,Uptrend_Weighted,Uptrend_LSTM], axis=1, sort=False)

result.to_csv('result.csv')


