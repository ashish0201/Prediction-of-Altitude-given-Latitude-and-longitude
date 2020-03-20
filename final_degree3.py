# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:54:50 2019

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

for i in range(0,temp_len-1):
    line=temp[i].split(",");
    for j in range(0,4):
        data[i][j]=float(line[j])
   
data=data[:,1:4]  
result=data[:,2:3]
data=data[:,0:2]

#Appending ones
data=np.append(arr=np.ones((len(data),1)),values=data,axis=1)#1
data=np.append(arr=data[:,1:2]*data[:,2:3],values=data,axis=1)#xy
data=np.append(arr=data[:,3:4]**2,values=data,axis=1)#y2
data=np.append(arr=data[:,3:4]**2,values=data,axis=1)#x2

data=np.append(arr=data[:,1:2]*data[:,5:6],values=data,axis=1)#y3
data=np.append(arr=data[:,2:3]*data[:,5:6],values=data,axis=1)#xy2
data=np.append(arr=data[:,2:3]*data[:,7:8],values=data,axis=1)#x2y
data=np.append(arr=data[:,3:4]*data[:,7:8],values=data,axis=1)#x3

traindata,testdata,trainresult,testresult=train_test_split(data,result,test_size=0.3)
len_of_terms=10
le=len_of_terms

avg=[0]*le
std=[0]*le
for i in range(0,le):
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

#-----------------------------
y_bar=0
for i in range(0,len(trainresult[:,0])):
    y_bar+=trainresult[i][0]
aa=np.std(trainresult[:,0])
y_bar/=len(trainresult[:,0])
sst=0
for i in range(0,len(trainresult[:,0])):
    sst+=(trainresult[i][0]-y_bar)**2
sst/=2
 
w=np.empty((1,le))
w.fill(0)
   #best-0.000002
error_per_iteration=np.zeros((300,1))
eta=8e-7
for iterate in range(0,300):
    print(iterate)
    print(w)
    sum_w=np.zeros((1,le))
    for row in range(0,len(traindata)):
        
        for j in range(0,le):
            sum_w[0][j]+=(traindata[row][j]*((np.dot(traindata[row],w.T))-trainresult[row][0]))
        

    for j in range(0,le):
        w[0][j]-=eta*(sum_w[0][j])

    
    error_sum=0
    for row in range(0,len(traindata)):
        error_sum+=(((np.dot(traindata[row],w.T))-trainresult[row][0])**2)
    error_sum/=2
    print(error_sum)
    error_per_iteration[iterate]=error_sum
    sse=error_sum
    r2=1-(sse/sst)
    print(r2)
    print("\n")

store_err=error_per_iteration
err=error_per_iteration[:20,:]
iteration_num = [(i*20) for i in range(0,20)]
plt.plot(iteration_num,err, color='g')
plt.xlabel('Number of iterations')
plt.ylabel('Error Loss')
plt.title('Gradient Descent Graph')
plt.show()