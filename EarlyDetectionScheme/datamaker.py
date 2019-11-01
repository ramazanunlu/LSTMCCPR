#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:07:11 2019

@author: ramazanunlu
"""

def GenerateData(w,t,abtype,normal_size,abnormal_size):
    import numpy as np
    import math
    import pandas as pd
    import itertools
    data2=[]
    data1=[]
    mu, sigma = 0, 1
    #generate normal data samples
    for i in range(normal_size):
        x=np.random.normal(mu, sigma, w)
        data1.append(x)
    #Generate abnormal data samples 
    for i in range(abnormal_size):
        if abtype ==1: #uptrend
            y=np.random.randn(w)+t*(np.arange(w)+1)
        
        elif abtype ==2: #downtrend
            y=np.random.randn(w)-t*(np.arange(w)+1)
 
        elif abtype ==3: #upshift
            y= np.random.randn(w)+ t*np.ones(w)
       
        elif abtype ==4: 
            y= np.random.randn(w)- t*np.ones(w) #downshift
         
        elif abtype ==5: 
            y= np.random.randn(w)+ t*(-1)**(np.arange(w)+1) #Systematic patterns   
            
        elif abtype ==6: 
            y= np.random.randn(w)+ t*np.cos(2*math.pi*(np.arange(w)+1)/8) #Cyclic
        
        elif abtype ==7: 
            y=t*np.random.randn(w) #Stratification
         
        
        data2.append(y)
        
    
    data1=np.array(data1)
    data2=np.array(data2)
     
     
    data=np.concatenate((data1, data2), axis=0)
    
    labels=np.concatenate((np.ones(normal_size), np.zeros(abnormal_size)), axis=0)
    labels=labels.reshape(-1,1)
    
    Final_Data=np.concatenate((data, labels), axis=1)
    '''
    Final_Data=pd.DataFrame(Final_Data)
    
    L1= list(itertools.repeat([1/normal_size], normal_size))
    L2= list(itertools.repeat([1/abnormal_size], abnormal_size))
    L=L1+L2
    L=pd.DataFrame(L)
    Final_Data=pd.concat([Final_Data,L],axis=1)
    '''
    return Final_Data

'''
plt.plot(data[949,:])
plt.plot(data[950,:])
'''