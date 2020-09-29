# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:09:43 2020

@author: ram
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
import math


#%%
abtype=5
normalsize=950
abnormalsize=50
w=30
j=w
alpha=np.arange(0.005,1.805,0.02)
All_ATPRL=[]
for ii in alpha:   
    df=makedata.GenerateData(w,ii,abtype,normalsize,abnormalsize)
    data=df
    classes=df[:,w]
    Attributes=df[:,0:w]
    ACC=[]
    Sensitivity=[]
    Specificity=[]
    Gmean=[]
    train_l=[]
    test_l=[]
    ATPRLIDs=[]
    ATPRLs=[]
    ROTs=[]
    
    for i in range(10):
        b=0
        c=0
        ROC=np.zeros([2,2])
        r=np.random.permutation(classes.size)
        tot= math.floor(classes.size*0.9)
        train=r[0:tot]
        train_l=classes[r[0:tot]]
        
        Data=[]
        test_l=[]
        
        for jj in range(10):
            Datax=[]
            AA=np.random.permutation(950)
            iq=AA[2]
            Data1=data[iq,0:w]
            Data1=Data1.reshape(1,-1)
            BB=np.random.permutation(50)+950
            ps=BB[2]
            Data2=data[ps,0:w]
            Data2=Data2.reshape(1,w)
            Datax.append(Data2)
            for ss in range(1,w+1):
                D=[]
                for j in range(ss,w):
                    D.append(Data1[0,j])
                for k in range(0,ss):
                   
                    D.append(Data2[0,k])
                Datax.append(np.array(D).reshape(1,w))
            Data.append(Datax)
            aaa=np.array(Data[0][0])
            for inx in range(len(Data)):
                for inx2 in range(len(Data[0])):
                    
                    aaa=np.concatenate((aaa, np.array(Data[inx][inx2])), axis=0)
            aaa = np.delete(aaa, (0), axis=0)
            
            test_l.append(np.zeros([w+1,1]))
            test_l[jj][0]=1
        bbb=np.array(test_l).reshape(310,1)
        
        clf = svm.SVC(kernel='poly',C=1,degree=4)
        clf.fit(Attributes[train,:], train_l)
        predict_label=clf.predict(aaa).reshape(-1,1)
        acc = metrics.accuracy_score(bbb, predict_label)
        ACC.append(acc)
        
        
        s=0
        kk=s+w
    
        for mn in range(10):
            ATPRL=0
            if predict_label[s:kk+1].all()==np.ones([w+1,1]).all():
                
                ATPRL=0
                ROT=0.5
                ATPRLID=0
            for yy in range(1,w+2):
                #print(yy)
                if predict_label[s+yy-1]==0:
                    ATPRL=yy
                    ROT=1
                    ATPRLID=ATPRL/ROT
                    break
            s=kk+1
            kk=s+w
            if ATPRL!=0:
                ATPRLs.append(ATPRL)
                ROTs.append(ROT)
                ATPRLIDs.append(ATPRLID)
                
    Acc=np.mean(ACC)
                
    if not ATPRL:
        ATPRL=float("NaN")
        ROT=float("NaN")
        ATPRLID=float("NaN")
    else:
        ATPRL=np.mean(ATPRLs)
        ROT=np.sum(ROTs)/100
        ATPRLID=ATPRL/ROT
    
    All_ATPRL.append(ATPRLID)  
    
    
    
    
                        
