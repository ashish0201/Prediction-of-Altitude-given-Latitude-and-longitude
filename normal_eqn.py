# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:46:27 2019

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D  
file = open("3D_spatial_network.txt",'r')
dataset = file.read()
temp=dataset.split("\n")
temp_len=len(temp)
data = np.zeros((temp_len,4))

#avg=np.zeros((1,4))
for i in range(0,temp_len-1):
    line=temp[i].split(",");
    #print (line)
    for j in range(0,4):
        data[i][j]=float(line[j])


data=data[:,1:4]  
result=data[:,2:3]
data=data[:,0:2]

data=np.append(arr=np.ones((len(data),1)),values=data,axis=1)#1


        
traindata,testdata,trainresult,testresult=train_test_split(data,result,test_size=0.3,random_state=0)

avg=[0]*3
std=[0]*3
for i in range(0,3):
    avg[i]=np.mean(traindata[:,i:i+1])
    std[i]=np.std(traindata[:,i:i+1])
    if std[i]!=0:
        traindata[:,i]=(traindata[:,i]-avg[i])/std[i]
        testdata[:,i]=(testdata[:,i]-avg[i])/std[i]

for i in range(0,1):
    avg[i]=np.mean(trainresult[:,i:i+1])
    std[i]=np.std(trainresult[:,i:i+1])
    if std[i]!=0:
        trainresult[:,i]=(trainresult[:,i]-avg[i])/std[i]
        testresult[:,i]=(testresult[:,i]-avg[i])/std[i]


X=traindata
Y=trainresult
temp=np.linalg.inv(np.dot(X.T,X))
temp2=np.dot(X.T,Y)
theta=np.dot(temp,temp2)

w=theta
w=np.array(w)
#w=w.T
X=testdata
Y=testresult
predict=np.zeros((1,len(X)))
for row in range(0,len(X)):
    predict[0][row]=(np.dot(X[row],w))
predict=np.transpose(predict)

y_bar=0
for i in range(0,len(Y[:,0])):
    y_bar+=Y[i][0]
y_bar/=len(Y[:,0])
sst=0
for i in range(0,len(Y[:,0])):
    sst+=(Y[i][0]-y_bar)**2
sst/=2

sse=0
for i in range(0,len(Y[:,0])):
    sse+=(Y[i][0]-predict[i][0])**2
sse/=2
r2=1-(sse/sst)
print(r2)
print("\n")


rms=(sse/len(Y[:,0]))**0.5

print(rms)
print("\n")

# =============================================================================
#r2--
# 0.02373103405942123

# rms=(sse/len(Y[:,0]))**0.5
# 
# rms
# Out[40]: 0.6986662171382642
# =============================================================================

