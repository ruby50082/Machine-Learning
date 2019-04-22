import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
iris = pd.read_csv('reviews1.csv')
iris2 = pd.read_csv('reviews2.csv')
iris3 = pd.read_csv('reviews3.csv')

iris3["s"].hist(bins=20, alpha=0.5)
iris2["sp"].hist(bins=20, alpha=0.5)
iris2["sj"].hist(bins=20, alpha=0.5)
iris["num"].hist(bins=20, alpha=0.5)
#iris["App"].hist(bins=2000, alpha=0.5)
#iris["Sentiment_Subjectivity"].hist(bins=20, alpha=0.5)


bins4 = [0, 40,80,120,160,200,240]


#Sentiment = pd.cut(iris['Sentiment'], bins)
num = pd.cut(iris['num'], bins4)
#Sentiment_Subjectivity = pd.cut(iris['Sentiment_Subjectivity'], bins1)

def get_stats(group):
    return {'count': group.count()}

grouped = iris['num'].groupby(num)
print(grouped)
bin_counts4 = grouped.apply(get_stats).unstack()
print bin_counts4
bin_counts4.plot(kind='bar')
plt.savefig('demo.png',bbox_inches='tight')


bin_counts = iris3['s']
print bin_counts
bin_counts.index = ['Positive', 'Neutral','Negative', 'nan']


bin_counts1 = iris2['sp']
print bin_counts1
bin_counts1.index = ['0.0', '(0.0,0.1]', '(0.1,0.2]', '(0.2,0.3]', '(0.3,0.4]', '(0.4,0.5]', '(0.5,0.6]', '(0.6,0.7]', '(0.7,0.8]', '(0.8,0.9]', '(0.9,1.0]','1.0']


bin_counts2 = iris2['sj']
print bin_counts2
bin_counts2.index = ['0.0', '(0.0,0.1]', '(0.1,0.2]', '(0.2,0.3]', '(0.3,0.4]', '(0.4,0.5]', '(0.5,0.6]', '(0.6,0.7]', '(0.7,0.8]', '(0.8,0.9]', '(0.9,1.0]','1.0']


'''
#bin_counts.index.name = 'Sentiment'
bin_counts.plot(kind='bar')
plt.savefig('demo.png',bbox_inches='tight')
'''
'''
#def get_stats(group):
#    return {'count': group.count()}
'''
#grouped1 = iris['myname']
#print(grouped1)



#bin_counts1 = iris['num']
#print bin_counts1

#bin_counts1.index1 = ['1.0~1.5', '1.5~2.0', '2.0~2.5', '2.5~3.0', '3.0~3.5', '3.5~4.0', '4.0~4.5', '4.5~5.0']
#bins1 = iris['myname']
bin_counts.index.name = 'Sentiment'
bin_counts.plot(kind='bar')
plt.savefig('demo1.png',bbox_inches='tight')


bin_counts1.index.name = 'Sentiment Polarity'
bin_counts1.plot(kind='bar')
plt.savefig('demo2.png',bbox_inches='tight')


bin_counts2.index.name = 'Sentiment Subjectivity'
bin_counts2.plot(kind='bar')
plt.savefig('demo3.png',bbox_inches='tight')

'''

def get_stats(group):
    return {'count': group.count()}

grouped2 = iris['Sentiment_Subjectivity'].groupby(Sentiment_Subjectivity)
print(grouped2)
bin_counts2 = grouped2.apply(get_stats).unstack()
print bin_counts2

#bin_counts.index2 = ['0.5~1.0', '1.0~1.5', '1.5~2.0', '2.0~2.5', '2.5~3.0', '3.0~3.5', '3.5~4.0', '4.0~4.5', '4.5~5.0', '5.0~5.5', '5.5~6.0', '6.0~6.5', '6.5~7.0',
#                    '7.0~7.5', '7.5~8.0']
bin_counts2.index.name = 'Sentiment_Subjectivity'
bin_counts2.plot(kind='bar')
plt.savefig('demo2.png',bbox_inches='tight')

'''