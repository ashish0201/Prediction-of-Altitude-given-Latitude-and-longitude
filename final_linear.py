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
data=np.append(arr=np.ones((len(data),1)),values=data,axis=1)
traindata,testdata,trainresult,testresult=train_test_split(data,result,test_size=0.3)

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
 
w=np.empty((1,3))
w.fill(50)
   #best-0.000002
error_per_iteration=np.zeros((300,1))
eta=2e-6
for iterate in range(0,300):
    print(iterate)
    print(w)
    sum_w=np.zeros((1,3))
    for row in range(0,len(traindata)):

        sum_w[0][0]+=(traindata[row][0]*((np.dot(traindata[row],w.T))-trainresult[row][0]))
        sum_w[0][1]+=(traindata[row][1]*((np.dot(traindata[row],w.T))-trainresult[row][0]))
        sum_w[0][2]+=(traindata[row][2]*((np.dot(traindata[row],w.T))-trainresult[row][0]))


    w[0][0]-=eta*(sum_w[0][0])
    w[0][1]-=eta*(sum_w[0][1])
    w[0][2]-=eta*(sum_w[0][2])
    
    error_sum=0
    for row in range(0,len(traindata)):
        #error_sum+=(((w[0][0]*traindata[row][0]+w[0][1]*traindata[row][1]+w[0][2]*traindata[row][2])-trainresult[row][0])**2)
        #error_sum+=(np.dot(traindata[row],w.T))**2
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
    
