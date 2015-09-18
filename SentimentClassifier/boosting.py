'''
Created on Sep 12, 2015

@author: ananta
'''
from sklearn.ensemble import AdaBoostClassifier
from sentiment_data import getTwitterData, getTfidfData, SimpleTimer, getPickeledData, outputScores
from sklearn.tree import DecisionTreeClassifier
import numpy

def tryVariousHyperParams(dataTrain, dataTest, train_tfidf, test_tfidf, outFile):
    params = [(15,500),(20,500),(25,800),(10,8000),(12,3000),(8,3000),(4,5000)]
    res = []
    for depth, num in params:
        outFile.write('depth %d, num %d \n'%(depth, num))
        with SimpleTimer('time to train', outFile):
            estimator = DecisionTreeClassifier(max_depth=depth)
            clf = AdaBoostClassifier(base_estimator=estimator,  n_estimators=num)
            clf.fit(train_tfidf, dataTrain.target)
            
        score = clf.score(test_tfidf, dataTest.target)
        print depth, num
        print score
        outFile.write('score %d \n'%(score))
        res.append((score, depth, num))
    res = sorted(res, key=lambda x:x[0])
    return res[0]

def runBoosting(dataTrain, dataTest, holdOut, train_tfidf, test_tfidf, hold_tfidf):
    outFile = open('boostingLog.txt','a')
    print 'running boosting'
    outFile.write('train==> %d, %d \n'%(train_tfidf.shape[0],train_tfidf.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_tfidf.shape[0],test_tfidf.shape[1]))
    # takes a very long time to run
#     score, bestDepth, num = tryVariousHyperParams(dataTrain, dataTest, train_tfidf, test_tfidf)
    bestDepth = 4
    bestNum = 5000
    with SimpleTimer('time to train', outFile):
        estimator = DecisionTreeClassifier(max_depth=bestDepth)
        bestClf = AdaBoostClassifier(base_estimator=estimator,  n_estimators=bestNum)
        bestClf.fit(train_tfidf, dataTrain.target)
    
    bestScore = bestClf.score(test_tfidf, dataTest.target)
    print 'the best score %.3f' % bestScore
    outFile.write('depth %d, num %d score %.3f \n'%(bestDepth, bestNum, bestScore))
    bestClf.fit(train_tfidf, dataTrain.target)
    predicted = bestClf.predict(test_tfidf)
    
    train_predict = bestClf.predict(train_tfidf)
    
    predicted = bestClf.predict(test_tfidf)
    print 'testing score'
    outFile.write('testing score')
    outputScores(dataTest.target, predicted, outFile)
    print 'training score'
    outFile.write('testing score')
    outputScores(dataTrain.target, train_predict, outFile)
    
    results = predicted == dataTest.target
    print numpy.mean(results)
    res = []
    for i in range(len(results)):
        if not results[i]:
            res.append(i)
    
    
if __name__ == '__main__':
    dataSize = 100
    dataTrain, dataTest, holdOut, train_tfidf, test_tfidf, hold_tfidf = getPickeledData()
    runBoosting(dataTrain, dataTest, holdOut, train_tfidf, test_tfidf, hold_tfidf)
