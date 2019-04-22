import numpy as np

from sklearn import tree

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

#import pydot

import pandas as pd 

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

#import matplotlib as mpl

#mpl.use('Agg')

#import matplotlib.pyplot as plt

from collections import Counter

#import seaborn as sns


########################### iris ################################

iris = pd.read_csv('iris.csv', header=None)

iris_data = iris.as_matrix(columns=iris.columns[0:4])

iris_target = iris.as_matrix(columns=iris.columns[4:5])

iris_target = iris_target.ravel()

feature_names = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']

le = preprocessing.LabelEncoder()

le.fit(iris_target)

target_names = list(le.classes_)

x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size = 0.3, random_state = 0 )

#feature_names0 = feature_names[0:2]
feature_names1 = feature_names[1:3]
feature_names2 = feature_names[2:4]
feature_names3 = feature_names[0:3:2]
#feature_names4 = feature_names[1:4:2]
#feature_names5 = feature_names[0:4:3]

#x0_train = x_train[:, 0:2]
x1_train = x_train[:, 1:3]
x2_train = x_train[:, 2:4]
x3_train = x_train[:, 0:3:2]
#x4_train = x_train[:, 1:4:2]
#x5_train = x_train[:, 0:4:3]

#x0_test = x_test[:, 0:2]
x1_test = x_test[:, 1:3]
x2_test = x_test[:, 2:4]
x3_test = x_test[:, 0:3:2]
#x4_test = x_test[:, 1:4:2]
#x5_test = x_test[:, 0:4:3]

y_train = y_train.ravel()
y_test = y_test.ravel()

#clf0 = tree.DecisionTreeClassifier()        #score = 0.6444444444444445
#clf0 = clf0.fit(x0_train, y_train)
clf1 = tree.DecisionTreeClassifier()         #score = 0.9555555555555556
clf1 = clf1.fit(x1_train, y_train)
clf2 = tree.DecisionTreeClassifier()         #score = 0.9555555555555556
clf2 = clf2.fit(x2_train, y_train)
clf3 = tree.DecisionTreeClassifier()         #score = 0.9333333333333333
clf3 = clf3.fit(x3_train, y_train)
#clf4 = tree.DecisionTreeClassifier()        #score = 0.8444444444444444
#clf4 = clf4.fit(x4_train, y_train)
#clf5 = tree.DecisionTreeClassifier()        #score = 0.8888888888888888
#clf5 = clf5.fit(x5_train, y_train)


############# the graph of desicion trees#########################
'''
dot_data = tree.export_graphviz(clf1, out_file=None, 
                         feature_names=feature_names1,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('iris_t1.png')

dot_data = tree.export_graphviz(clf2, out_file=None, 
                         feature_names=feature_names2,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('iris_t2.png')

dot_data = tree.export_graphviz(clf3, out_file=None, 
                         feature_names=feature_names3,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('iris_t3.png')
'''
##################################################################

pred=["" for x in range(len(y_test))]
for i in range(len(y_test)):
    result0=clf1.predict([x1_test[i,:]])
    result1=clf2.predict([x2_test[i, 0:4]])
    result2=clf3.predict([x3_test[i,0 :4]])

    if(np.array_equal(result0,result1)):
        pred[i]=result0
    elif (np.array_equal(result1,result2)):
        pred[i]=result1
    elif(np.array_equal(result0,result1)):
        pred[i]=result0
    else:
        pred[i]=result0

match=0

for i in range(len(pred)):   
    if(y_test[i] in pred[i]):
        match=match+1
print("iris_data:\n")
print("random_forest:", float(match)/float(len(pred)))

######### k-fold ##########

kf = KFold(5,True,1)
n=kf.get_n_splits(iris_data)

trainset=["" for x in range(120)]
traintar=["" for x in range(120)]
testset=["" for x in range(30)]
testtar=["" for x in range(30)]

total_sum=0
for i,j in kf.split(iris_data) :
    for x in range(len(i)):
        trainset[x]=iris_data[i[x]]
        traintar[x]=iris_target[i[x]]
        
    for y in range(len(j)):
        testset[y]=iris_data[j[y]]
        testtar[y]=iris_target[j[y]]
        
    clf4= tree.DecisionTreeClassifier()
    clf4= clf4.fit(trainset, traintar)     
    result=clf4.predict(testset)
    match=(result==testtar)
    total_sum=total_sum+match.sum()

print("k-fold:",float((total_sum/5))/float(len(testset)))

######## confusion matrix ##########

print("(confusion matrix:")
print( confusion_matrix(y_test,pred))
print(")")
print("accuracy of confusion matrix:",accuracy_score(y_test,pred))


################## the graph of k-ford ############################
'''
uniform_data = pd.DataFrame(confusion_matrix(y_test,pred))
sns_plot = sns.heatmap(uniform_data)
fig = sns_plot.get_figure()
fig.savefig('demo.png',bbox_inches='tight')
'''
###################################################################


######### resubstitution ###########

total_sum2=0
clf5 = tree.DecisionTreeClassifier()
clf5 = clf5.fit(iris_data, iris_target)
result2=clf5.predict(iris_data)
match2=(result2==iris_target)
print("resubstitution:",float(match2.sum())/float(len(iris_data)))
print("\n\n")

###################google-play-reviews###################

review = pd.read_csv('googleplaystore_user_reviews.csv',delimiter=',')

review=review.fillna(method ='ffill')

review_target = review.as_matrix(columns=review.columns[2:3])

r_le = preprocessing.LabelEncoder()

for col in review.columns:
     review[col] = r_le.fit_transform(review[col]) 

review_data = review.as_matrix()

review_data = np.delete(review_data, 2, 1)

review_target = review_target.ravel()

feature_names = list(review)

r_le.fit(review_target)

target_names = list(r_le.classes_)

x_train, x_test, y_train, y_test = train_test_split(review_data, review_target, test_size = 0.3, random_state = 0 )

#feature_names0 = feature_names[0:1]
feature_names1 = feature_names[2:3]
#feature_names2 = feature_names[3:4]
feature_names3 = feature_names[0:3:2]
feature_names4 = feature_names[2:4:3]
#feature_names5 = feature_names[0:4:3]

#x0_train = x_train[:, 0:1]
x1_train = x_train[:, 2:3]
#x2_train = x_train[:, 3:4]
x3_train = x_train[:, 0:3:2]
x4_train = x_train[:, 2:4:3]
#x5_train = x_train[:, 0:4:3]

#x0_test = x_test[:, 0:1]
x1_test = x_test[:, 2:3]
#x2_test = x_test[:, 3:4]
x3_test = x_test[:, 0:3:2]
x4_test = x_test[:, 2:4:3]
#x5_test = x_test[:, 0:4:3]

y_train = y_train.ravel()
y_test = y_test.ravel()

#clf0 = tree.DecisionTreeClassifier()    #score = 0.7289128518844937
#clf0 = clf0.fit(x0_train, y_train)
clf1 = tree.DecisionTreeClassifier()    #score = 0.9999481569806625
clf1 = clf1.fit(x1_train, y_train)
#clf2 = tree.DecisionTreeClassifier()    #score = 0.8052257763492146
#clf2 = clf2.fit(x2_train, y_train)
clf3 = tree.DecisionTreeClassifier()    #score = 0.9999481569806625
clf3 = clf3.fit(x3_train, y_train)
clf4 = tree.DecisionTreeClassifier()    #score = 0.9999481569806625
clf4 = clf4.fit(x4_train, y_train)
#clf5 = tree.DecisionTreeClassifier()    #score = 0.8583130281507595
#clf5 = clf5.fit(x5_train, y_train)

################## the graph of decison trees ####################
'''

dot_data = tree.export_graphviz(clf1, out_file=None, 
                         feature_names=feature_names1,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('revuew_t1.png')

dot_data = tree.export_graphviz(clf3, out_file=None, 
                         feature_names=feature_names3,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('review_t2.png')

dot_data = tree.export_graphviz(clf4, out_file=None, 
                         feature_names=feature_names4,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('review_t3.png')
'''
####################################################################

####### pred ##############
pred=["" for x in range(len(y_test))]
for i in range(len(y_test)):
    #xx1_test=np.reshape(x1_test,(-1,2))
    result1=clf1.predict([x1_test[i,:]])
    result3=clf3.predict([x3_test[i,:]])
    result4=clf4.predict([x4_test[i,:]])

    if(np.array_equal(result1,result3)):
        pred[i]=result1
    elif (np.array_equal(result1,result4)):
        pred[i]=result1
    elif(np.array_equal(result3,result4)):
        pred[i]=result3
    else:
        pred[i]=result1
        
match=0

for i in range(len(pred)):   
    if(y_test[i] in pred[i]):
        match=match+1
print("googleplaystore_user_review_data:\n")
print("random_forest:",float(match)/float(len(pred)))   

############### k-fold ##################

kf = KFold(167,True,1)

#trainset=["" for x in range(58450)]  #64295
#traintar=["" for x in range(58450)]
#testset=["" for x in range(5845)]
#testtar=["" for x in range(5845)]
trainset=["" for x in range(63910)]  #64295
traintar=["" for x in range(63910)]
testset=["" for x in range(385)]
testtar=["" for x in range(385)]
total_sum=0
for i,j in kf.split(review_data) :
    for x in range(len(i)):
        trainset[x]=review_data[i[x]]
        traintar[x]=review_target[i[x]]
        
    for y in range(len(j)):
        testset[y]=review_data[j[y]]
        testtar[y]=review_target[j[y]]
        
    clf_k= tree.DecisionTreeClassifier()
    clf_k= clf_k.fit(trainset, traintar)     
    result=clf_k.predict(testset)
    match=(result==testtar)
    total_sum=total_sum+match.sum()

print("k-fold:",float(total_sum/167)/float(len(testset)))

######## confusion matrix ##########

print("(confusion matrix:")
print(confusion_matrix(y_test,pred))
print(")")
print("accuracy of confusion matrix:",accuracy_score(y_test,pred))

############## the graph of k-ford ###########################
'''
uniform_data = pd.DataFrame(confusion_matrix(y_test,pred))
sns_plot = sns.heatmap(uniform_data)
fig = sns_plot.get_figure()
fig.savefig('demo1.png',bbox_inches='tight')
'''
#############################################################

######### resubstitution ###########

clf_r = tree.DecisionTreeClassifier()
clf_r = clf_r.fit(review_data, review_target)
result_r=clf_r.predict(review_data)
match_r=(result_r==review_target)
print("resubstitution:",float(match_r.sum())/float(len(review_data)))
print("\n\n")


####################### google ########################

#get_ipython().run_line_magic('matplotlib', 'inline')

df=pd.read_csv('googleplaystore.csv')

feature_names = list(df)
df=df.drop_duplicates(subset=None, keep='first', inplace=False)

#create a month list to find wrong data in "Last Updated"
month=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

#get the update_date colnum from dataframe
update_date=df['Last Updated']

#seperate the  date string into three colnum including month, day, and year
date_terms = pd.DataFrame(update_date.str.split('\s|,\s').tolist())
wrongline=[]

#find the line number with the wrong data in "Last Updated". This results from merging of two colnum data into a term
for i in range(len(date_terms[0])):
    if (date_terms[0][i] not in month):
        wrongline.append(i)
        
#delete the line with the wrong data in "Last Updated"

df=df.drop(df.index[wrongline])

#change date format

df['Last Updated']=pd.to_datetime(df['Last Updated'], dayfirst=True)

df['Price'] = df['Price'].str.replace('$', '')

df['Installs'] = df['Installs'].str.replace('+', '')
df['Installs'] = df['Installs'].str.replace(',', '')

df['Type'] = df['Type'].str.replace('Free', '0')
df['Type'] = df['Type'].str.replace('Paid', '1')

df['Size'] = df['Size'].str.replace('Varies with device', 'NaN')
df.Size = (df.Size.replace(r'[kM]+$', '', regex=True).astype(float) *  df.Size.str.extract(r'[\d\.]+([kM]+)', expand=False)
                .fillna(1)
                .replace(['k','M'], [10**3, 10**6]).astype(int))
mean_Size=df['Size'].mean()

df['Size'].fillna(mean_Size,inplace=True)

df['Reviews']=pd.to_numeric(df['Reviews'], errors='coerce')
#mean_Reviews=df['Reviews'].mean()

df['Installs']=pd.to_numeric(df['Installs'], errors='coerce')
#mean_Installs=df['Installs'].mean()

df['Price']=pd.to_numeric(df['Price'], errors='coerce')
df['Type']=pd.to_numeric(df['Type'], errors='coerce')
df['Type'].fillna(0,inplace=True)
#mean_Price=df['Price'].mean()

df['Category']=df['Category'].str.capitalize()
df['Category']=df['Category'].str.replace('_and_', ' & ')
df['Genres']=df['Genres'].str.capitalize()

genres=df['Genres']
genres_terms = pd.DataFrame(genres.str.split(';').tolist())
genres_terms_row_length=genres_terms.count(axis='columns')

m=len(df)

for i in range(len(genres_terms[0])):
    if (genres_terms_row_length[:][i] > 1):
        df.iat[i,9]=genres_terms[0][i]
        for j in range(genres_terms_row_length[:][i]-1):
            k=j+1
            df = df.append(df.iloc[i],ignore_index=True)
            df.iat[m,9]=genres_terms[k][i]
            m=m+1
            
df['Genres']=df['Genres'].str.capitalize()

mean_Rating=df['Rating'].mean()

df['Rating'].fillna(mean_Rating,inplace=True)


df['Content Rating'] = df['Content Rating'].str.replace('10', '')
df['Content Rating'] = df['Content Rating'].str.replace('Teen', 'Children')
df['Content Rating'] = df['Content Rating'].str.replace('17', '')
df['Content Rating'] = df['Content Rating'].str.replace('s', '')
df['Content Rating'] = df['Content Rating'].str.replace('only', '')
df['Content Rating'] = df['Content Rating'].str.replace('18', '')
df['Content Rating'] = df['Content Rating'].str.replace('+', '')

df['Content Rating'] = df['Content Rating'].str.replace('Everyone', '3')
df['Content Rating'] = df['Content Rating'].str.replace('Children', '0')
df['Content Rating'] = df['Content Rating'].str.replace('Mature', '1')
df['Content Rating'] = df['Content Rating'].str.replace('Adult', '2')

df['Content Rating']=pd.to_numeric(df['Content Rating'], errors='coerce')

df['Content Rating'].fillna(3,inplace=True)

df['Android Ver'].fillna('1.0.0 and up',inplace=True)
df['Android Ver'] = df['Android Ver'].str.replace('Varies with device', '1.0.0 and up')
df['Android Ver'] = df['Android Ver'].str.split('\s').str[0]
df['Android Ver'] = df['Android Ver'].str.slice(start=0, stop=3)
df['Android Ver'] = df['Android Ver'].str.replace('.', '', case=False, regex=False)
df['Android Ver']=pd.to_numeric(df['Android Ver'], errors='coerce')

df['Current Ver'] = df['Current Ver'].str.replace('Varies with device', '1')
df['Current Ver'] = df['Current Ver'].str.replace('\s', '', case=False, regex=False)
df['Current Ver'] = df['Current Ver'].str.split('.').str[0]
df['Current Ver']=pd.to_numeric(df['Current Ver'], errors='coerce')
df['Current Ver'].fillna(1,inplace=True)

df=df.drop('Current Ver', axis=1)


###########################################################################################################################
#plot features

##### IRIS #####
'''
iris["sepal length"].hist(bins=20, alpha=0.5)
iris["sepal width"].hist(bins=20, alpha=0.5)
iris["petal length"].hist(bins=20, alpha=0.5)
iris["petal width"].hist(bins=20, alpha=0.5)

bins = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
bins1 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
bins2 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
bins3 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

sepal_length = pd.cut(iris['sepal length'], bins)
sepal_width = pd.cut(iris['sepal width'], bins1)
petal_length = pd.cut(iris['petal length'], bins2)
petal_width = pd.cut(iris['petal width'], bins3)

def get_stats(group):
    return {'count': group.count()}

grouped = iris['sepal length'].groupby(sepal_length)
print(grouped)
bin_counts = grouped.apply(get_stats).unstack()
print bin_counts

bin_counts.index.name = 'sepal length'
bin_counts.plot(kind='bar')
plt.savefig('demo.png',bbox_inches='tight')

def get_stats(group):
    return {'count': group.count()}
grouped1 = iris['sepal width'].groupby(sepal_width)
print(grouped1)
bin_counts1 = grouped1.apply(get_stats).unstack()
print bin_counts1

bin_counts1.index.name = 'sepal width'
bin_counts1.plot(kind='bar')
plt.savefig('demo1.png',bbox_inches='tight')

def get_stats(group):
    return {'count': group.count()}

grouped2 = iris['petal length'].groupby(petal_length)
print(grouped2)
bin_counts2 = grouped2.apply(get_stats).unstack()
print bin_counts2

bin_counts2.index.name = 'petal length'
bin_counts2.plot(kind='bar')
plt.savefig('demo2.png',bbox_inches='tight')


def get_stats(group):
    return {'count': group.count()}

grouped3 = iris['petal width'].groupby(petal_width)
print(grouped3)
bin_counts3 = grouped3.apply(get_stats).unstack()
print bin_counts3

bin_counts3.index.name = 'petal width'
bin_counts3.plot(kind='bar')
plt.savefig('demo3.png',bbox_inches='tight')

'''
##### google-play-reviews #####
'''
reviews = pd.read_csv('1.csv')

reviews["num"].hist(bins=20, alpha=0.5)
bin_counts = reviews['num']
bin_counts.index = ['Positive', 'Neutral','Negative', 'nan']
bin_counts.plot(kind='bar')
plt.savefig('demo.png',bbox_inches='tight')

reviews["num1"].hist(bins=20, alpha=0.5)
bin_counts = reviews['num1']
bin_counts.index = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, -1]
bin_counts.plot(kind='bar')
plt.savefig('demo1.png',bbox_inches='tight')

reviews["num2"].hist(bins=20, alpha=0.5)
bin_counts = reviews['num2']
bin_counts.index = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, -1]
bin_counts.plot(kind='bar')
plt.savefig('demo2.png',bbox_inches='tight')

reviews["App"].hist(bins=2000, alpha=0.5)
bins = [0, 40, 80, 120, 160, 200, 240, 280, 320]
App = pd.cut(reviews['App'], bins)
def get_stats(group):
    return {'count': group.count()}

bin_counts.plot(kind='bar')
plt.savefig('demo3.png',bbox_inches='tight')
'''
##### google-play-store #####

###'Category'

#labels, values = zip(*Counter(df['Category']).items())
#indexes = np.arange(len(labels))
#width = 0.8
#plt.bar(indexes, values, width, color='b')
#plt.xticks(indexes - width * 0.5, labels)
#plt.xticks(rotation='vertical')
#plt.title('Bar Plot of Category', fontsize=18)
#plt.ylabel('Quantity', fontsize=16)
#plt.show()

###'Rating'

#df_sort_Rating=df.sort_values(by=['Rating'])
#plt_sort=df_sort_Rating['Rating']
#labels, values = zip(*Counter(plt_sort).items())
#avg=np.mean(plt_sort)
#stdv=np.std(plt_sort)
#num_bins=len(set(plt_sort))+1
#plt.hist(plt_sort, bins=num_bins, alpha=0.5, facecolor='g', align='mid')

#plt.text(3, 2300, r'$\mu\ =%.2f$' % avg, fontdict={'size': 12, 'color': 'b'})
#plt.text(3, 2150, r'$\sigma\ =%.2f$' %stdv, fontdict={'size': 12, 'color': 'b'})
#plt.title('Histogram of Rating', fontsize=18)
#plt.xlabel('Rating', fontsize=16)
#plt.ylabel('Quantity', fontsize=16)
#plt.show()


###'Reviews'

#df_sort_Reviews=df.sort_values(by=['Reviews'])
#plt_sort=df_sort_Reviews['Reviews']
#labels, values = zip(*Counter(plt_sort).items())
#avg=np.mean(plt_sort)
#stdv=np.std(plt_sort)
#num_bins=len(set(plt_sort))+1
#plt.hist(plt_sort, bins=num_bins, alpha=0.5, facecolor='g', align='mid')
#plt.text(4e7, 70, r'$\mu\ =%.2f$' % avg, fontdict={'size': 12, 'color': 'b'})
#plt.text(4e7, 65, r'$\sigma\ =%.2f$' %stdv, fontdict={'size': 12, 'color': 'b'})
#plt.title('Histogram of Reviews', fontsize=18)
#plt.xlabel('Reviews', fontsize=16)
#plt.ylabel('Quantity', fontsize=16)
#plt.ylim(0, 90)
#plt.show()

###'Size'

#df_sort=df.sort_values(by=['Size'])
#plt_sort=df_sort['Size']
#labels, values = zip(*Counter(plt_sort).items())
#avg=np.mean(plt_sort)
#stdv=np.std(plt_sort)
#num_bins=len(set(plt_sort))+1
#plt.hist(plt_sort, bins=num_bins, alpha=0.5, facecolor='g', align='mid')
#plt.text(6e7, 1500, r'$\mu\ =%.2f$' % avg, fontdict={'size': 12, 'color': 'b'})
#plt.text(6e7, 1400, r'$\sigma\ =%.2f$' %stdv, fontdict={'size': 12, 'color': 'b'})
#plt.title('Histogram of Size', fontsize=18)
#plt.xlabel('Size', fontsize=16)
#plt.ylabel('Quantity', fontsize=16)
#plt.show()

###'Installs'

#df_sort=df.sort_values(by=['Installs'])
#plt_sort=df_sort['Installs']
#labels, values = zip(*Counter(plt_sort).items())
#avg=np.mean(plt_sort)
#stdv=np.std(plt_sort)
#num_bins=len(set(plt_sort))+1
#plt.hist(plt_sort, bins=num_bins, alpha=0.5, facecolor='g', align='mid')
#plt.text(6e8, 9000, r'$\mu\ =%.2f$' % avg, fontdict={'size': 12, 'color': 'b'})
#plt.text(6e8, 8000, r'$\sigma\ =%.2f$' %stdv, fontdict={'size': 12, 'color': 'b'})
#plt.title('Histogram of Installs', fontsize=18)
#plt.xlabel('Installs', fontsize=16)
#plt.ylabel('Quantity', fontsize=16)
#plt.show()

###'Type'
#df_sort=df.sort_values(by=['Type'])
#plt_sort=df_sort['Type']

#labels, values = zip(*Counter(plt_sort).items())
#indexes = np.arange(len(labels))
#width = 0.8
#plt.bar(indexes, values, width, color='b')
#plt.xticks(indexes - width * 0.5, labels)
#plt.xticks(rotation='vertical')
#plt.title('Bar Plot of Type', fontsize=18)
#plt.ylabel('Quantity', fontsize=16)
#plt.xticks([0, 1], ['Free', 'Paid'], rotation='horizontal', fontsize=16) 
#plt.xlim(-1, 2)
#plt.show()

###'Price'

#df_sort=df.sort_values(by=['Price'])
#plt_sort=df_sort['Price']
#labels, values = zip(*Counter(plt_sort).items())
#avg=np.mean(plt_sort)
#stdv=np.std(plt_sort)
#num_bins=len(set(plt_sort))+1
#plt.hist(plt_sort, bins=num_bins, alpha=0.5, facecolor='g', align='mid')
#plt.text(250, 9000, r'$\mu\ =%.2f$' % avg, fontdict={'size': 12, 'color': 'b'})
#plt.text(250, 8000, r'$\sigma\ =%.2f$' %stdv, fontdict={'size': 12, 'color': 'b'})
#plt.title('Histogram of Price', fontsize=18)
#plt.xlabel('Price', fontsize=16)
#plt.ylabel('Quantity', fontsize=16)
#plt.show()

###'Content Rating'
#df_sort=df.sort_values(by=['Content Rating'])
##plt_sort=df_sort['Content Rating']

#labels, values = zip(*Counter(plt_sort).items())
#indexes = np.arange(len(labels))
#width = 0.8
#plt.bar(indexes, values, width, color='b')
#plt.xticks(indexes - width * 0.5, labels)
#plt.xticks(rotation='vertical')
#plt.title('Bar Plot of Content Rating', fontsize=18)
#plt.ylabel('Quantity', fontsize=16)
#plt.xticks([0, 1, 2, 3], ['Children', 'Mature', 'Adult', 'Everyone'], rotation='horizontal', fontsize=16) 
#plt.xlim(-1, 4)
#plt.show()

###'Genres'

#labels, values = zip(*Counter(df['Genres']).items())
#indexes = np.arange(len(labels))
#plt.bar(indexes, values, width, color='b')
#plt.xticks(indexes - width * 0.5, labels, fontsize=7)
#plt.xticks(rotation='vertical')
#plt.title('Bar Plot of Genres', fontsize=18)
#plt.ylabel('Quantity', fontsize=16)
#plt.show()

###'Last Updated' 

#df_sort=df.sort_values(by=['Last Updated'])
#plt_sort=df_sort['Last Updated']
#labels, values = zip(*Counter(plt_sort).items())

#num_bins=len(set(plt_sort))+1
#plt.hist(plt_sort, bins=num_bins, alpha=0.5, facecolor='g', align='mid')
#plt.title('Histogram of Last Updated', fontsize=18)
#plt.xlabel('Last Updated (Time)', fontsize=16)
#plt.ylabel('Quantity', fontsize=16)
#plt.show()

###'Android Ver' 

#df_sort=df.sort_values(by=['Android Ver'])
#plt_sort=df_sort['Android Ver']
#labels, values = zip(*Counter(plt_sort).items())

#num_bins=len(set(plt_sort))+1
#plt.hist(plt_sort, bins=num_bins, alpha=0.5, facecolor='g', align='mid')
#plt.title('Histogram of Android Ver', fontsize=18)
#plt.xlabel('Android Ver', fontsize=16)
#plt.ylabel('Quantity', fontsize=16)
#plt.xticks([10, 20, 30, 40, 50], ['1.0', '2.0', '3.0', '4.0', '5.0'], rotation='horizontal', fontsize=16) 
#plt.xlim(0, 60)
#plt.show()

###########################################################################################################################


category={'Family': '0', 'Game': '1', 'Tools': '2', 'Business': '3', 'Medical': '4',
          'Productivity': '5', 'Personalization': '6', 'Lifestyle': '7', 'Communication': '8',
          'Finance': '9', 'Sports': '10', 'Photography': '11', 'Health & fitness': '12', 'Social': '13',
          'News & magazines': '14', 'Travel & local': '15', 'Books & reference': '16', 'Shopping': '17', 'Dating': '18',
          'Video_players': '19', 'Education': '20', 'Maps & navigation': '21', 'Food & drink': '22', 'Entertainment': '23',
          'Auto & vehicles': '24', 'Libraries & demo': '25', 'Weather': '26', 'House & home': '27', 'Parenting': '28',
          'Art & design': '29', 'Events': '30', 'Comics': '31', 'Beauty': '32'}

df['Category'] = df['Category'].map(category)
df['Category']=pd.to_numeric(df['Category'], errors='coerce')

genres={'Tools': '2', 'Education': '20', 'Entertainment': '23', 'Business': '3', 'Medical': '4', 'Productivity': '5',
        'Personalization': '6', 'Lifestyle': '7', 'Action': '0', 'Sports': '10', 'Communication': '8', 'Finance': '9',
        'Photography': '11', 'Health & fitness': '12', 'Social': '13', 'News & magazines': '14', 'Casual': '1',
        'Travel & local': '15', 'Arcade': '19', 'Books & reference': '16', 'Shopping': '17', 'Simulation': '33',
        'Dating': '18', 'Video players & editors': '34', 'Puzzle': '35', 'Maps & navigation': '21', 'Action & adventure': '36',
        'Food & drink': '22', 'Role playing': '37', 'Racing': '38', 'Strategy': '39', 'Educational': '40', 'Adventure': '41',
        'Auto & vehicles': '24', 'Libraries & demo': '25', 'Weather': '26', 'House & home': '27', 'Pretend play': '42',
        'Art & design': '29', 'Brain games': '43', 'Events': '30', 'Board': '44', 'Comics': '31', 'Parenting': '28',
        'Beauty': '32', 'Card': '45', 'Music & video': '46', 'Trivia': '47', 'Casino': '48', 'Creativity': '49', 'Word': '50',
        'Music': '51', 'Music & audio': '52'}
df['Genres'] = df['Genres'].map(genres)
df['Genres']=pd.to_numeric(df['Genres'], errors='coerce')

y=df['Content Rating']
y=y.values.ravel()

df=df.drop('Content Rating', axis=1)
df=df.drop('App', axis=1)
df=df.drop('Last Updated', axis=1)

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

#############

x_train=x_train.values #convert df to array
x_test=x_test.values

x0_train = x_train[:, [0,4]]
x1_train = x_train[:, [0, 7]]
x2_train = x_train[:, [4, 7]]
x3_train = x_train[:, [0, 4, 7]]
x4_train = x_train[:, [0, 6]]

x5_train = x_train[:, [0, 4, 7]]
x6_train = x_train[:, [0, 6, 7]]
#x7_train = x_train[:, [0, 1, 7]]
x7_train = x_train[:, [0, 4, 6, 7]]

x0_test = x_test[:, [0, 4]]
x1_test = x_test[:, [0, 7]]
x2_test = x_test[:, [4, 7]]
x3_test = x_test[:, [0, 4, 7]]
x4_test = x_test[:, [0, 6]]

x5_test = x_test[:, [0, 4, 7]]
x6_test = x_test[:, [0, 6, 7]]
#x7_test = x_test[:, [0, 1,  7]]
x7_test = x_test[:, [0, 4, 6, 7]]

y_train = y_train.ravel()
y_test = y_test.ravel()


clf0 = tree.DecisionTreeClassifier()        
clf0 = clf0.fit(x0_train, y_train)
#print("score0:", clf0.score(x0_test,y_test))

clf1 = tree.DecisionTreeClassifier()         
clf1 = clf1.fit(x1_train, y_train)
#print("score1:", clf1.score(x1_test,y_test))

clf2 = tree.DecisionTreeClassifier()        
clf2 = clf2.fit(x2_train, y_train)
#print("score2:", clf2.score(x2_test,y_test))

clf3 = tree.DecisionTreeClassifier()        
clf3 = clf3.fit(x3_train, y_train)
#print("score3:", clf3.score(x3_test,y_test))

clf4 = tree.DecisionTreeClassifier()         
clf4 = clf4.fit(x4_train, y_train)
#print("score4:", clf4.score(x4_test,y_test))

clf5 = tree.DecisionTreeClassifier()       
clf5 = clf5.fit(x5_train, y_train)
#print("score5:", clf5.score(x5_test,y_test))


clf6 = tree.DecisionTreeClassifier()        
clf6 = clf6.fit(x6_train, y_train)
#print("score6:", clf6.score(x6_test,y_test))

clf7 = tree.DecisionTreeClassifier()        
clf7 = clf7.fit(x7_train, y_train)
#print("score7:", clf7.score(x7_test,y_test))


#clf8 = tree.DecisionTreeClassifier()        
#clf8 = clf8.fit(x8_train, y_train)
#print("score8:", clf8.score(x8_test,y_test))
'''
dot_data = tree.export_graphviz(clf0, out_file=None, 
                         feature_names=feature_names0,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('google_t0.png')

dot_data = tree.export_graphviz(clf1, out_file=None, 
                         feature_names=feature_names1,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('google_t1.png')

dot_data = tree.export_graphviz(clf2, out_file=None, 
                         feature_names=feature_names2,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('google_t2.png')

dot_data = tree.export_graphviz(clf3, out_file=None, 
                         feature_names=feature_names3,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('google_t3.png')

dot_data = tree.export_graphviz(clf4, out_file=None, 
                         feature_names=feature_names4,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('google_t4.png')

dot_data = tree.export_graphviz(clf5, out_file=None, 
                         feature_names=feature_names5,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('google_t5.png')

dot_data = tree.export_graphviz(clf6, out_file=None, 
                         feature_names=feature_names6,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('google_t6.png')

dot_data = tree.export_graphviz(clf7, out_file=None, 
                         feature_names=feature_names7,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('google_t7.png')

'''
pred=["" for x in range(len(y_test))]
result=["" for x in range(8)]

for i in range(len(y_test)):
    result[0]=clf0.predict([x0_test[i, :]])
    result[1]=clf1.predict([x1_test[i, :]])
    result[2]=clf2.predict([x2_test[i, :]])
    result[3]=clf3.predict([x3_test[i, :]])
    result[4]=clf4.predict([x4_test[i, :]])
    result[5]=clf5.predict([x5_test[i, :]])
    result[6]=clf6.predict([x6_test[i, :]])
    result[7]=clf7.predict([x7_test[i, :]])
    Children=0
    Mature=0
    Adult=0
    Everyone=0
    for j in range(len(result)):
        if(result[j]==0):
            Children=Children+1
        elif(result[j]==1):
            Mature=Mature+1
        elif(result[j]==2):
            Adult=Adult+1
        elif(result[j]==3):
            Everyone=Everyone+1
        

    if(Children>=Mature and Children>= Adult and Children>=Everyone):
        pred[i]=0
    elif(Mature>=Children and Mature>=Adult and Mature>=Everyone):
        pred[i]=1
    elif(Adult>=Children and  Adult>= Mature and  Adult>=Everyone):
        pred[i]=2
    else:
        pred[i]=3
    

match1=0
for i in range(len(pred)):
    if(y_test[i] ==pred[i]):
        match1=match1+1
print("googleplaystore:\n")
print("random_forest:",float(match1)/float(len(pred)))

############### k-fold ##################
from sklearn.model_selection import KFold

kf = KFold(349,True,1) #10819

trainset=["" for x in range(10788)]
traintar=["" for x in range(10788)]
testset=["" for x in range(31)]
testtar=["" for x in range(31)]
df=df.values
total_sum=0
for i,j in kf.split(df) :
    for a in range(len(i)):
        trainset[a]=df[i[a]]
        traintar[a]=y[i[a]]
        
    for b in range(len(j)):
        testset[b]=df[j[b]]
        testtar[b]=y[j[b]]
        
    
    clf9= tree.DecisionTreeClassifier()
    clf9= clf9.fit(trainset, traintar)     
    result=clf9.predict(testset)
    match=(result==testtar)
    total_sum=total_sum+match.sum()

print("k-fold:",float(total_sum/349)/float(len(testset)))


#################### the graph of k-ford ####################
'''
uniform_data = pd.DataFrame(confusion_matrix(y_test,pred))
sns_plot = sns.heatmap(uniform_data)
fig = sns_plot.get_figure()
fig.savefig('demo.png',bbox_inches='tight')
'''
#############################################################


######## confusion matrix ##########

from sklearn.metrics import confusion_matrix
pred=np.asarray(pred)

print("(confusion matrix:")
print(confusion_matrix(y_test,pred))
print(")")
print("accuracy of confusion matrix:",accuracy_score(y_test,pred))

######### resubstitution ###########

clf10 = tree.DecisionTreeClassifier()
clf10= clf10.fit(df, y)
result2=clf10.predict(df)
match2=(result2==y)
print("resubstitution:",float(match2.sum())/float(len(df)))

