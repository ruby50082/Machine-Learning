#!/usr/bin/env python
# coding: utf-8

# In[28]:


# degree=1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data1.csv')

z=df['sqft']
for i in range(z.shape[0]):
    if z[i] > 1:
        z[i]=1

x = df.iloc[:,1:18].values  # 1:18
y = df.iloc[:,0].values

x = (x - np.amin(x, axis=0))/(np.amax(x, axis=0)-np.amin(x, axis=0)) 
y = (y - np.amin(y, axis=0))/(np.amax(y, axis=0)-np.amin(y, axis=0)) 

x=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
y=y[:,np.newaxis]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[29]:


def MSE (X, Y, W):
    return np.sum((Y - np.dot(X, W))**2)/(2*X.shape[0])

def R2 (mse, num, y):
    return 1-((mse*num)/(np.sum((y - np.mean(y))**2)))


# In[31]:


learning_rate=2
iterations=10000
epsilon=0.00005

theta = np.random.randn(18,1)


x_train = np.array(x_train, dtype=np.float128)
for i in range(iterations):
    for j in range(18):
        predict = np.dot(x_train, theta)
        gradients = np.dot((y_train - predict).T, x_train[:,j])/x_train.shape[0]
        theta[j,0] += (learning_rate*gradients)

    err = MSE(x_train, y_train, theta)
    r2 = R2(err, x_train.shape[0], y_train)

    if err < epsilon:
        break
    print("iter %d, mse= %.6f, r2= %.6f" % (i,err,r2))  
    
err = MSE(x_test, y_test, theta)
r2 = R2(err, x_test.shape[0], y_test)

print("\ntesting:\nmse= %.6f, r2= %.6f" % (err,r2))


# In[32]:


test_data=[[0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
           [0, 2/18, 2.0, 1.0, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
test_data=np.asarray(test_data)

test_data[:,5:17]=(test_data[:,5:17] - np.amin(test_data[:,5:17], axis=0))/(np.amax(test_data[:,5:17], axis=0)-np.amin(test_data[:,5:17], axis=0)) 

test_data=np.concatenate((np.ones((test_data.shape[0],1)),test_data),axis=1)
predict = np.dot(test_data, theta)
#print(predict)
num=0
for num in range(14):
    if num == 0:
        print ("empty house: ",predict[num])
        
    if num == 1:
        print ("only with internet: ",predict[num])
        
    if num == 2:
        print ("only with air-conditioning: ",predict[num])
        
    if num == 3:
        print ("only with television: ",predict[num])
        
    if num == 4:
        print ("only with Satellite/Cable: ",predict[num])
        
    if num == 5:
        print ("only with heater: ",predict[num])
        
    if num == 6:
        print ("only with washing machine: ",predict[num])
        
    if num == 7:
        print ("only with elevator: ",predict[num])
        
    if num == 8:
        print ("only with kitchen: ",predict[num])
        
    if num == 9:
        print ("only with hot tub: ",predict[num])
        
    if num == 10:
        print ("only with fitness room: ",predict[num])
        
    if num == 11:
        print ("only with concierge: ",predict[num])
        
    if num == 12:
        print ("only with parking: ",predict[num])
        
    if num == 13:
        print ("internet & air-conditioning: ",predict[num])
        


# In[33]:


# degree=2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data1.csv')

z=df['sqft']
for i in range(z.shape[0]):
    if z[i] > 1:
        z[i]=1

x = df.iloc[:,1:18].values
y = df.iloc[:,0].values

x = (x - np.amin(x, axis=0))/(np.amax(x, axis=0)-np.amin(x, axis=0)) 
y = (y - np.amin(y, axis=0))/(np.amax(y, axis=0)-np.amin(y, axis=0)) 

tmp1=x**2
for i in range(1,16):
    for j in range(i+1,17):
        tmp2=x[:,i]*x[:,j]
        tmp1=np.concatenate((tmp1,tmp2[:,np.newaxis]),axis=1)
        
x=np.concatenate((np.ones((x.shape[0],1)),x,tmp1),axis=1)

x=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
y=y[:,np.newaxis]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[34]:


learning_rate=2
iterations=1000
epsilon=0.00005

theta = np.random.randn(156,1)


x_train = np.array(x_train, dtype=np.float128)
for i in range(iterations):
    for j in range(156):
        predict = np.dot(x_train, theta)
        gradients = np.dot((y_train - predict).T, x_train[:,j])/x_train.shape[0]
        theta[j,0] += (learning_rate*gradients)

    err = MSE(x_train, y_train, theta)
    r2 = R2(err, x_train.shape[0], y_train)

    if err < epsilon:
        break
    print("iter %d, mse= %.6f, r2= %.6f" % (i,err,r2))  
    
err = MSE(x_test, y_test, theta)
r2 = R2(err, x_test.shape[0], y_test)

print("\ntesting:\nmse= %.6f, r2= %.6f" % (err,r2))


# In[35]:


test_data=[[0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
           [0, 2/18, 2.0, 1.0, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
test_data=np.asarray(test_data)

test_data[:,5:17]=(test_data[:,5:17] - np.amin(test_data[:,5:17], axis=0))/(np.amax(test_data[:,5:17], axis=0)-np.amin(test_data[:,5:17], axis=0)) 


tmp3=test_data**2
for i in range(0,15):
    for j in range(i+1,16):
        tmp4=test_data[:,i]*test_data[:,j]
        tmp3=np.concatenate((tmp3,tmp4[:,np.newaxis]),axis=1)
test_data=np.asarray (test_data)
test_data=np.concatenate((np.ones((test_data.shape[0],1)),test_data,tmp3),axis=1)

test_data=np.concatenate((np.ones((test_data.shape[0],1)),test_data),axis=1)
predict = np.dot(test_data, theta)
num=0
for num in range(14):
    if num == 0:
        print ("empty house: ",predict[num])
        
    if num == 1:
        print ("only with internet: ",predict[num])
        
    if num == 2:
        print ("only with air-conditioning: ",predict[num])
        
    if num == 3:
        print ("only with television: ",predict[num])
        
    if num == 4:
        print ("only with Satellite/Cable: ",predict[num])
        
    if num == 5:
        print ("only with heater: ",predict[num])
        
    if num == 6:
        print ("only with washing machine: ",predict[num])
        
    if num == 7:
        print ("only with elevator: ",predict[num])
        
    if num == 8:
        print ("only with kitchen: ",predict[num])
        
    if num == 9:
        print ("only with hot tub: ",predict[num])
        
    if num == 10:
        print ("only with fitness room: ",predict[num])
        
    if num == 11:
        print ("only with concierge: ",predict[num])
        
    if num == 12:
        print ("only with parking: ",predict[num])
        
    if num == 13:
        print ("internet & air-conditioning: ",predict[num])
        


# In[36]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data1.csv')

z=df['sqft']
for i in range(z.shape[0]):
    if z[i] > 1:
        z[i]=1
        
x = df.iloc[:,1:18].values  # 1:18
y = df.iloc[:,0].values
mini=np.amin(y, axis=0)
maxi=np.amax(y, axis=0)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# create a Linear Regressor   
lin_regressor = LinearRegression()

# pass the order of your polynomial here  
poly = PolynomialFeatures(1)

# convert to be used further to linear regression
X_transform = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
# fit this to Linear Regressor
lin_regressor.fit(X_transform,y_train) 

# get the predictions
y_preds = lin_regressor.predict(X_test)
print(lin_regressor.score(X_transform,y_train))
print(lin_regressor.score(X_test,y_test))


# In[37]:


test_data=[[0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
           [0, 2/18, 2.0, 1.0, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
X_transform = poly.fit_transform(test_data)
predict=y_preds*(maxi-mini)+mini
num=0
for num in range(14):
    if num == 0:
        print ("empty house:\n",predict[num])
        
    if num == 1:
        print ("only with internet:\n",predict[num])
        
    if num == 2:
        print ("only with air-conditioning:\n",predict[num])
        
    if num == 3:
        print ("only with television:\n",predict[num])
        
    if num == 4:
        print ("only with Satellite/Cable:\n",predict[num])
        
    if num == 5:
        print ("only with heater:\n",predict[num])
        
    if num == 6:
        print ("only with washing machine:\n",predict[num])
        
    if num == 7:
        print ("only with elevator:\n",predict[num])
        
    if num == 8:
        print ("only with kitchen:\n",predict[num])
            
    if num == 9:
        print ("only with hot tub:\n",predict[num])
        
    if num == 10:
        print ("only with fitness room:\n",predict[num])
        
    if num == 11:
        print ("only with concierge:\n",predict[num])
        
    if num == 12:
        print ("only with parking:\n",predict[num])
        
    if num == 13:
        print ("internet & air-conditioning:\n",predict[num])


# In[38]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data1.csv')

z=df['sqft']
for i in range(z.shape[0]):
    if z[i] > 1:
        z[i]=1
        
x = df.iloc[:,1:18].values  # 1:18
y = df.iloc[:,0].values
mini=np.amin(y, axis=0)
maxi=np.amax(y, axis=0)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# create a Linear Regressor   
lin_regressor = LinearRegression()

# pass the order of your polynomial here  
poly = PolynomialFeatures(2)

# convert to be used further to linear regression
X_transform = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
# fit this to Linear Regressor
lin_regressor.fit(X_transform,y_train) 

# get the predictions
y_preds = lin_regressor.predict(X_test)
print(lin_regressor.score(X_transform,y_train))
print(lin_regressor.score(X_test,y_test))


# In[39]:


test_data=[[0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
           [0, 2/18, 2.0, 1.0, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
X_transform = poly.fit_transform(test_data)
predict=y_preds*(maxi-mini)+mini
num=0
for num in range(14):
    if num == 0:
        print ("empty house:\n",predict[num])
        
    if num == 1:
        print ("only with internet:\n",predict[num])
        
    if num == 2:
        print ("only with air-conditioning:\n",predict[num])
        
    if num == 3:
        print ("only with television:\n",predict[num])
        
    if num == 4:
        print ("only with Satellite/Cable:\n",predict[num])
        
    if num == 5:
        print ("only with heater:\n",predict[num])
        
    if num == 6:
        print ("only with washing machine:\n",predict[num])
        
    if num == 7:
        print ("only with elevator:\n",predict[num])
        
    if num == 8:
        print ("only with kitchen:\n",predict[num])
            
    if num == 9:
        print ("only with hot tub:\n",predict[num])
        
    if num == 10:
        print ("only with fitness room:\n",predict[num])
        
    if num == 11:
        print ("only with concierge:\n",predict[num])
        
    if num == 12:
        print ("only with parking:\n",predict[num])
        
    if num == 13:
        print ("internet & air-conditioning:\n",predict[num])

