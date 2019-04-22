import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
iris = pd.read_csv('iris1.csv')

iris["sepal length"].hist(bins=20, alpha=0.5)
iris["sepal width"].hist(bins=20, alpha=0.5)
iris["petal length"].hist(bins=20, alpha=0.5)
iris["petal width"].hist(bins=20, alpha=0.5)
iris["flower name"].hist(bins=20, alpha=0.5)

bins = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
bins1 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
bins2 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
bins3 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
bins4 = [0,1,2,3]

sepal_length = pd.cut(iris['sepal length'], bins)
sepal_width = pd.cut(iris['sepal width'], bins1)
petal_length = pd.cut(iris['petal length'], bins2)
petal_width = pd.cut(iris['petal width'], bins3)
flower_name = pd.cut(iris['flower name'], bins4)

def get_stats(group):
    return {'count': group.count()}

grouped = iris['sepal length'].groupby(sepal_length)
print(grouped)
bin_counts = grouped.apply(get_stats).unstack()
print bin_counts

#bin_counts.index = ['3.5~4.0', '4.0~4.5', '4.5~5.0', '5.0~5.5', '5.5~6.0', '6.0~6.5', '6.5~7.0',
 #                   '7.0~7.5', '7.5~8.0', '8.0~8.5']
bin_counts.index.name = 'sepal length'
bin_counts.plot(kind='bar')
plt.savefig('demo.png',bbox_inches='tight')

def get_stats(group):
    return {'count': group.count()}
grouped1 = iris['sepal width'].groupby(sepal_width)
print(grouped1)
bin_counts1 = grouped1.apply(get_stats).unstack()
print bin_counts1

#bin_counts1.index1 = ['1.0~1.5', '1.5~2.0', '2.0~2.5', '2.5~3.0', '3.0~3.5', '3.5~4.0', '4.0~4.5', '4.5~5.0']
bin_counts1.index.name = 'sepal width'
bin_counts1.plot(kind='bar')
plt.savefig('demo1.png',bbox_inches='tight')

def get_stats(group):
    return {'count': group.count()}

grouped2 = iris['petal length'].groupby(petal_length)
print(grouped2)
bin_counts2 = grouped2.apply(get_stats).unstack()
print bin_counts2

#bin_counts.index2 = ['0.5~1.0', '1.0~1.5', '1.5~2.0', '2.0~2.5', '2.5~3.0', '3.0~3.5', '3.5~4.0', '4.0~4.5', '4.5~5.0', '5.0~5.5', '5.5~6.0', '6.0~6.5', '6.5~7.0',
#                    '7.0~7.5', '7.5~8.0']
bin_counts2.index.name = 'petal length'
bin_counts2.plot(kind='bar')
plt.savefig('demo2.png',bbox_inches='tight')


def get_stats(group):
    return {'count': group.count()}

grouped3 = iris['petal width'].groupby(petal_width)
print(grouped3)
bin_counts3 = grouped3.apply(get_stats).unstack()
print bin_counts3

#bin_counts.index3 = ['0.5~1.0', '1.0~1.5', '1.5~2.0', '2.0~2.5', '2.5~3.0']
bin_counts3.index.name = 'petal width'
bin_counts3.plot(kind='bar')
plt.savefig('demo3.png',bbox_inches='tight')

def get_stats(group):
    return {'count': group.count()}

grouped4 = iris['flower name'].groupby(flower_name)
print(grouped4)
bin_counts4 = grouped4.apply(get_stats).unstack()
print bin_counts4

#bin_counts.index3 = ['0.5~1.0', '1.0~1.5', '1.5~2.0', '2.0~2.5', '2.5~3.0']
bin_counts4.index.name = 'flower name'
bin_counts4.plot(kind='bar')
plt.savefig('demo4.png',bbox_inches='tight')