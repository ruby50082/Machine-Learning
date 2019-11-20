#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor  
import  matplotlib.pyplot as plt
import graphviz 

df=pd.read_csv('data.csv')

tar=df['price']
data=df.drop('price', axis=1)
temp=df['sqft']
for i in range(temp.shape[0]):
    if temp[i]>1:
        temp[i]=1
target= (tar-min(tar))/(max(tar)-min(tar))

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 0 )

regr = DecisionTreeRegressor()
regr.fit(x_train,y_train)
print ("Training score:%f"%(regr.score(x_train,y_train)))
print ("Test score:%f"%(regr.score(x_test,y_test)))

###changing max depth
maxdepth=23
depths = np.arange(1,maxdepth)
training_scores = []
testing_scores = []
for depth in depths:
    regr = DecisionTreeRegressor(max_depth=depth)
    regr.fit(x_train,y_train)
    training_scores.append(regr.score(x_train,y_train))
    test_score=regr.score(x_test,y_test)
    testing_scores.append(test_score)
    print('depth',depth,  ' ',test_score) 
#plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(depths,training_scores,label='traing score')
ax.plot(depths,testing_scores,label='testing_scores')
ax.set_xlabel("maxdepth")
ax.set_ylabel("score")
ax.set_title("Decision Tree Regression")
ax.legend(framealpha=0.5)
plt.show()

####pca
for i in range(1,x_train.shape[1]):
    pca=PCA(n_components=i)
    pca.fit(data)
    pcatrain=pca.transform(x_train)
    pcatest= pca.transform(x_test)
    regressor = DecisionTreeRegressor()
    regreessor=regressor.fit(pcatrain, y_train)
    print(i,'components:  Test score= ' , regressor.score(pcatest,y_test))

###another application
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
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
fin=regr.predict(test_data)
fin=fin*(max(tar)-min(tar))+min(tar)
num=0
for num in range(14):
    if num == 0:
        print ("empty house: ", fin[num] )
        
    if num == 1:
        print ("only with internet: ", fin[num])
       
    if num == 2:
        print ("only with air-conditioning: ", fin[num])
       
    if num == 3:
        print ("only with television: ", fin[num])
        
    if num == 4:
        print ("only with Satellite/Cable: ", fin[num])
        
    if num == 6:
        print ("only with washing machine: ", fin[num])
       
    if num == 7:
        print ("only with elevator: ", fin[num])
      
    if num == 8:
        print ("only with kitchen: ", fin[num])

    if num == 9:
        print ("only with hot tub: ", fin[num])
       
    if num == 10:
        print ("only with fitness room: ", fin[num])
       
    if num == 11:
        print ("only with concierge: ", fin[num])
       
    if num == 12:
        print ("only with parking: ", fin[num])


# In[ ]:





# In[ ]:




