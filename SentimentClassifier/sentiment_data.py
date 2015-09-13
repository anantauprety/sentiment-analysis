'''
Created on Sep 12, 2015

@author: ananta
'''
from os import path
import re
import time
from sklearn.datasets.base import Bunch
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
import numpy

class SimpleTimer(object):

    def __init__(self, txt, file=None):
        self.text = txt or ''
        self.outFile = file

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        if self.outFile:
            self.outFile.write('%s %.3f \n' % (self.text, time.time() - self.start))
        print self.text, time.time() - self.start
        

class TwitterData(object):
    
    def __init__(self,filePath):
        self.names = ['negative','positive']
        self.filePath = filePath
        self.train, self.test = self.getData()
        
    def getData(self):
        res = []
        if path.exists(self.filePath):
            with open(self.filePath, 'r') as inFile:
                for _ in range(50000):
                    line = inFile.readline()
                    tar = int(line[0])
                    txt = re.sub(r'[^ \w]+', '', line[1:].strip())
                    txt = re.sub(r'[\d_]+', '', txt)
                    res.append((tar,txt.lower()))
            shuffle(res)
            size = len(res)
        print 'total data size %d' % len(res)
        train, test = res[:2*size/3], res[2*size/3:]
        return train, test
            
        
    def fetchData(self, subset='train', n_sample=10):
        if subset == 'train':
            return self.shuffleData(self.train[:n_sample])
        elif subset == 'test':
            return self.shuffleData(self.test[:n_sample])
            
    def shuffleData(self, res):
            shuffle(res)
            train = Bunch()
            train.data = map(lambda x:x[1], res)
            train.target = map(lambda x:x[0], res)
            train.target_names = self.names
            return train

def getTwitterData(size=10, ratio=0.2):
#     filePath = 'C:\\Users\\ananta\\Documents\\6220\\Sentiment-Analysis-Dataset\\TwitterSentiment.txt'
    filePath = 'TwitterSentiment.txt'
    twtrData = TwitterData(filePath)
    
    with SimpleTimer('time to fetch training data'):
        dataTrain = twtrData.fetchData(subset='train', n_sample=int(size-size*ratio))
        print '%.3f negative on training data' % (len(filter(lambda x:x == 0, dataTrain.target)) * 1.0 / len(dataTrain.target) * 100)
    with SimpleTimer('time to fetch testing data'):
        dataTest = twtrData.fetchData(subset='test', n_sample=int(size*ratio))
    return dataTrain, dataTest

def getTfidfData(dataTrain, dataTest):
    print dataTrain.target_names
    
    count_vect = CountVectorizer(strip_accents='ascii', stop_words='english', max_features=len(dataTrain.target) * 2)
    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    X_counts = count_vect.fit_transform(dataTrain.data)
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    print X_tfidf.shape
    
    Y_counts = count_vect.transform(dataTest.data)
    Y_tfidf = tfidf_transformer.transform(Y_counts)
    print Y_tfidf.shape
    
    print 'feature selection using chi square test'
    feature_names = count_vect.get_feature_names()
    
    ch2 = SelectKBest(chi2, k=len(dataTrain.target))
    X_tfidf = ch2.fit_transform(X_tfidf, dataTrain.target)
    Y_tfidf = ch2.transform(Y_tfidf)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
        
    if feature_names:
        feature_names = numpy.asarray(feature_names)
        print 'important features'
        print feature_names.shuffle()[:10]
    return X_tfidf, Y_tfidf
    
    
if __name__ == '__main__':
    from pprint import pprint
    filePath = 'TwitterSentiment.txt'
#     print TwitterData(filePath).fetchData('train',10)
    dataTrain, dataTest = getTwitterData(100)
#     pprint(dataTrain.data)
    getTfidfData(dataTrain, dataTest)
    
    