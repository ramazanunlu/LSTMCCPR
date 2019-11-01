#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:17:53 2019

@author: ramazanunlu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

path = 'data/'
 
files = os.listdir(path)
#files=['systematic_results.csv']
for name in files:
    filename_read=os.path.join(path,name)
    print(name)
    data=pd.read_csv(filename_read,delimiter=';')



    SVM_Gmean=data[['WindowLength','Parameter','G-Mean']]
    WSVM_Gmean=data[['WindowLength','Parameter','G-Mean.1']]
    LSTM_Gmean=data[['WindowLength','Parameter','G-Mean.2']]

    SVM_Specificity=data[['WindowLength','Parameter','Specificity']]
    WSVM_Specificity=data[['WindowLength','Parameter','Specificity.1']]
    LSTM_Specificity=data[['WindowLength','Parameter','Specificity.2']]

    SVM_Sensitivity=data[['WindowLength','Parameter','Sensitivity']]
    WSVM_Sensitivity=data[['WindowLength','Parameter','Sensitivity.1']]
    LSTM_Sensitivity=data[['WindowLength','Parameter','Sensitivity.2']]

    SVM_Accuracy=data[['WindowLength','Parameter','Accuracy']]
    WSVM_Accuracy=data[['WindowLength','Parameter','Accuracy.1']]
    LSTM_Accuracy=data[['WindowLength','Parameter','Accuracy.2']]

    SVM_Gmean= SVM_Gmean.pivot("WindowLength", "Parameter", "G-Mean")
    WSVM_Gmean= WSVM_Gmean.pivot("WindowLength", "Parameter", "G-Mean.1")
    LSTM_Gmean= LSTM_Gmean.pivot("WindowLength", "Parameter", "G-Mean.2")

    SVM_Specificity= SVM_Specificity.pivot("WindowLength", "Parameter", "Specificity")
    WSVM_Specificity= WSVM_Specificity.pivot("WindowLength", "Parameter", "Specificity.1")
    LSTM_Specificity= LSTM_Specificity.pivot("WindowLength", "Parameter", "Specificity.2")

    SVM_Sensitivity= SVM_Sensitivity.pivot("WindowLength", "Parameter", "Sensitivity")
    WSVM_Sensitivity= WSVM_Sensitivity.pivot("WindowLength", "Parameter", "Sensitivity.1")
    LSTM_Sensitivity= LSTM_Sensitivity.pivot("WindowLength", "Parameter", "Sensitivity.2")

    SVM_Accuracy= SVM_Accuracy.pivot("WindowLength", "Parameter", "Accuracy")
    WSVM_Accuracy= WSVM_Accuracy.pivot("WindowLength", "Parameter", "Accuracy.1")
    LSTM_Accuracy= LSTM_Accuracy.pivot("WindowLength", "Parameter", "Accuracy.2")

    fig, axn = plt.subplots(4,3,sharey=True,sharex=True,figsize=(20, 8))
    cbar_ax = fig.add_axes([0.6, 0.105, .01, 0.85])
    
    for i, ax in enumerate(axn.flat):
        if i == 0:
            sns.heatmap(SVM_Gmean, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
            ax.set_ylabel('G-mean')
            ax.set_xlabel('')
            ax.set_title('SVM')
        elif i ==1:
                sns.heatmap(WSVM_Gmean, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                ax.set_ylabel('')
                ax.set_xlabel('')
                ax.set_title('WSVM')
        elif i ==2:
                sns.heatmap(LSTM_Gmean, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                ax.set_ylabel('')
                ax.set_xlabel('')  
                ax.set_title('LSTM')
    
        elif i ==3:
                sns.heatmap(SVM_Specificity, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                ax.set_ylabel('Specificity')
                ax.set_xlabel('')
        elif i ==4:
                sns.heatmap(WSVM_Specificity, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                ax.set_ylabel('')
                ax.set_xlabel('')  
        elif i ==5:
                sns.heatmap(LSTM_Specificity, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                ax.set_ylabel('')
                ax.set_xlabel('') 
        elif i ==6:
                sns.heatmap(SVM_Sensitivity, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                ax.set_ylabel('Sensitivity')
                ax.set_xlabel('')
        elif i ==7:
                    sns.heatmap(WSVM_Sensitivity, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                    ax.set_ylabel('')
                    ax.set_xlabel('')  
        elif i ==8:
                    sns.heatmap(LSTM_Sensitivity, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                    ax.set_ylabel('')
                    ax.set_xlabel('') 
        elif i ==9:
                    sns.heatmap(SVM_Accuracy, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                    ax.set_ylabel('Accuracy')
                    ax.set_xlabel('')
        elif i ==10:
                    sns.heatmap(WSVM_Accuracy, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                    ax.set_ylabel('')
                    ax.set_xlabel('Parameters')  
                
        elif i ==11:
                    sns.heatmap(LSTM_Accuracy, ax=ax,cmap="YlGnBu",
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
                    ax.set_ylabel('')
                    ax.set_xlabel('') 
 
    fig.tight_layout(rect=[0, 0, .6, 1])       
#fig.text(0.000005, 0.03, 'Parameters', ha='center')
    fig.text(0.00001, 0.5, 'Window Length', va='center', rotation='vertical')
#fig.text(0.0000008, 0.89, 'G-mean', va='center', rotation='vertical')
#fig.text(0.0000008, 0.65, 'Specificity', va='center', rotation='vertical')
#fig.text(0.0000008, 0.3, 'Sensitivity', va='center', rotation='vertical')
#fig.text(0.0000008, 0.1, 'Accuracy', va='center', rotation='vertical')
    fig.savefig(name[0:-4],dpi=700,format='pdf')
#%% Levels graph
import datamaker as makedata
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
w=30
alpha=[0.155,0.33,0.505,0.68,0.855,1.03,1.205,1.38,1.555,1.805]#np.arange(0.005,1.806,0.025) #np.arange(0.005,1.805,0.025)
#alpha=[0.45]
y=np.random.randn(w)
data2= []
for i in alpha:
    data=y+i*(np.arange(w)+1)
    data2.append(data)
data2.append(y)
data2=pd.DataFrame(data2)
line,=plt.plot(data2.iloc[0,:],label='Level1')
line,=plt.plot(data2.iloc[1,:],label='Level2')
line,=plt.plot(data2.iloc[2,:],label='Level3')
line,=plt.plot(data2.iloc[3,:],label='Level4')
line,=plt.plot(data2.iloc[4,:],label='Level5')
line,=plt.plot(data2.iloc[5,:],label='Level6')
line,=plt.plot(data2.iloc[6,:],label='Level7')
line,=plt.plot(data2.iloc[7,:],label='Level8')
line,=plt.plot(data2.iloc[8,:],label='Level9')
line,=plt.plot(data2.iloc[9,:],label='Level10')
line,=plt.plot(data2.iloc[10,:],label='Normal',lw=3,ls='--')
plt.ylabel('Values')
plt.xlabel('Data samples')
plt.legend(loc=0,ncol=2)
plt.savefig('levels.png',dpi=500,format='png')
