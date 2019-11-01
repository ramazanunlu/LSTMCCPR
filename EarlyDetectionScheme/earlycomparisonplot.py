# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:41:21 2019

@author: ram
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data=pd.read_csv('Systematic_EarlyDetection.csv',sep=';')
data=data.drop(['Alpha'],axis=1)
data=data[['SVM_1','WSVM_1','LSTM_1']]

minus1=data[data==-1].count()
Zeros=data[data==0].count()
nanvalue=data.isnull().sum()
ontimeSVM=73-(minus1['SVM_1']+nanvalue['SVM_1']+Zeros['SVM_1'])
ontimeWSVM=73-(minus1['WSVM_1']+nanvalue['WSVM_1']+Zeros['WSVM_1'])
ontimeLSTM=73-(minus1['LSTM_1']+nanvalue['LSTM_1']+Zeros['LSTM_1'])

Names=['0','-1','-','On Time']
values_SVM=[Zeros['SVM_1'],minus1['SVM_1'],nanvalue['SVM_1'],ontimeSVM]
values_SVM=np.array(values_SVM).reshape(1,4)

values_WSVM=[Zeros['WSVM_1'],minus1['WSVM_1'],nanvalue['WSVM_1'],ontimeWSVM]
values_WSVM=np.array(values_WSVM).reshape(1,4)

values_LSTM=[Zeros['LSTM_1'],minus1['LSTM_1'],nanvalue['LSTM_1'],ontimeLSTM]
values_LSTM=np.array(values_LSTM).reshape(1,4)

final_data=np.concatenate((values_SVM,values_WSVM,values_LSTM))

plt.bar(Names, values)