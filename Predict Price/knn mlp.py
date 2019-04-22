import numpy as np
import pandas as pd 
from sklearn import neighbors
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')
x = data.drop(['price'],axis=1)
y = data['price']

for i in range(y.shape[0]):
    if y[i] < 100:
        y[i]=0
    elif y[i]>=100 and y[i]<150:
        y[i]=1
    elif y[i]>=150 and y[i]<200:
        y[i]=2
    elif y[i]>=200 and y[i]<250:
        y[i]=3
    elif y[i]>=250 and y[i]<300:
        y[i]=4
    elif y[i]>=300 and y[i]<350:
        y[i]=5
    elif y[i]>=350 and y[i]<400:
        y[i]=6
    elif y[i]>=400 and y[i]<450:
        y[i]=7
    elif y[i]>=450 and y[i]<500:
        y[i]=8
    elif y[i]>=500 and y[i]<550:
        y[i]=9
    elif y[i]>=550 and y[i]<600:
        y[i]=10
    else:
        y[i]=11

x=np.asarray(x)
y=np.asarray(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )

##### knn #####

from sklearn import neighbors
clf1 = neighbors.KNeighborsClassifier(1)
clf1.fit(x_train, y_train)
print("max score: ",clf1.score(x_test, y_test))

# most value amenetly #####
#          kind sqft bed bath rate internet air TV cable heater washing elevator kitchen hot tub fit concierge park
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
fin=clf1.predict(test_data)
#print (clf1.predict(test_data))
num=0
for num in range(14):
    if num == 0:
        print ("empty house: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 1:
        print ("only with internet: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 2:
        print ("only with air-conditioning: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 3:
        print ("only with television: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 4:
        print ("only with Satellite/Cable: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 5:
        print ("only with heater: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 6:
        print ("only with washing machine: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 7:
        print ("only with elevator: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 8:
        print ("only with kitchen: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 9:
        print ("only with hot tub: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 10:
        print ("only with fitness room: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 11:
        print ("only with concierge: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 12:
        print ("only with parking: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 13:
        print ("internet & air-conditioning: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")

##### plot accuracy & error rate #####

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
X_train=x_train
X_test=x_test
error = []
score = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    sc = knn.score(x_test, y_test)
    #print (sc)
    score.append(sc)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), score, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('score')  
plt.xlabel('K Value')  
plt.ylabel('score')  

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  

##### mlp #####

#print (x_train.shape)
X_train=x_train
X_test=x_test
from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(100,100,100))
mlp = MLPClassifier(solver='adam',hidden_layer_sizes=(17,500,13))
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

##### print training history #####

from sklearn.metrics import classification_report,confusion_matrix
#print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

##### most value amenety #####

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
clf1=mlp
fin=clf1.predict(test_data)
#print (clf1.predict(test_data))
num=0
for num in range(14):
    if num == 0:
        print ("empty house: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 1:
        print ("only with internet: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 2:
        print ("only with air-conditioning: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 3:
        print ("only with television: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 4:
        print ("only with Satellite/Cable: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 5:
        print ("only with heater: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 6:
        print ("only with washing machine: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 7:
        print ("only with elevator: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 8:
        print ("only with kitchen: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 9:
        print ("only with hot tub: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 10:
        print ("only with fitness room: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 11:
        print ("only with concierge: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 12:
        print ("only with parking: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")
    if num == 13:
        print ("internet & air-conditioning: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 150")
        elif fin[num] ==2:
            print("150 ~ 200")
        elif fin[num]==3:
            print("200 ~ 250")
        elif fin[num] ==4:
            print("250 ~ 300")
        elif fin[num]==5:
            print("300 ~ 350")
        elif fin[num] ==6:
            print("350 ~ 400")
        elif fin[num]==7:
            print("400 ~ 450")
        elif fin[num] ==8:
            print("450 ~ 500")
        elif fin[num]==9:
            print("500 ~ 550")
        elif fin[num] ==10:
            print("550 ~ 600")
        elif fin[num]==11:
            print("> 600")

#print (mlp.n_layers_)
#print (mlp.n_iter_)
#print (mlp.loss_)
#print (mlp.out_activation_)
#len(mlp.coefs_[0])
#len(mlp.intercepts_[0])