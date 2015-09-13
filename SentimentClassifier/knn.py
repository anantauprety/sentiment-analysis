'''
Created on Sep 12, 2015

@author: ananta
'''

from sklearn.neighbors import KNeighborsClassifier
from sentiment_data import getTwitterData, getTfidfData, SimpleTimer
from plot_learning_curve import plot_learning_curve
import numpy

def runKNNSimulation(dataTrain, dataTest, train_tfidf, test_tfidf):
    outFile = open('knnLog.txt','a')
    
    outFile.write('train==> %d, %d \n'%(train_tfidf.shape[0],train_tfidf.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_tfidf.shape[0],test_tfidf.shape[1]))
    with SimpleTimer('time to train', outFile):
        clf = KNeighborsClassifier(weights='distance', ).fit(train_tfidf, dataTrain.target)
    
    baseScore = clf.score(test_tfidf, dataTest.target)
    baseParams = clf.get_params(True)
    baseNeighbors = baseParams['n_neighbors']
    print 'baseline score %.3f base n_neighbors %d' % (baseScore, baseNeighbors)
    outFile.write('baseline score %.3f base height %d \n' % (baseScore, baseNeighbors))
    
    res = []
    with SimpleTimer('time to fine tune number of neighbors', outFile):
        for neighbors in range(2,200):
#             print 'training for neighbors %d' % neighbors
            clf = KNeighborsClassifier(n_neighbors=neighbors, weights='distance').fit(train_tfidf, dataTrain.target)
            score = clf.score(test_tfidf, dataTest.target)
            res.append((score, neighbors))
            outFile.write('%d %.3f \n' % (neighbors, score))
    res = sorted(res, key=lambda x:x[0], reverse=True)
    print res[:5]
    bestNeighbors = res[0][1]
    print ('best number of neighbors is %d' % bestNeighbors)
    outFile.write('best number of neighbors is %d  and score is %d\n' % (bestNeighbors, res[0][0]))
    bestClf = KNeighborsClassifier(n_neighbors=bestNeighbors, weights='distance')
    bestClf.fit(train_tfidf, dataTrain.target)
    predicted = bestClf.predict(test_tfidf)
    
    results = predicted == dataTest.target
    print numpy.mean(results)
    res = []
    for i in range(len(results)):
        if not results[i]:
            res.append(i)
    print 'classifier got these wrong:'
    for i in res[:10]:
        print dataTest.data[i], dataTest.target[i]
        outFile.write('%s %d \n' % (dataTest.data[i], dataTest.target[i]))
    '''
    train_sizes, train_scores, valid_scores = learning_curve(DecisionTreeClassifier(), train_tfidf, dataTrain.target, train_sizes=[50, 80, 110], cv=5)
    print train_sizes
    print train_scores
    print valid_scores
    '''
#     plot_learning_curve(KNeighborsClassifier(n_neighbors=25, weights='uniform', algorithm='kd_tree'), 'knn', X_tfidf, dataTrain.target, cv=5, n_jobs=4)
       
#     plot_learning_curve(bestClf, 'knn with %d neighbors' % bestNeighbors, train_tfidf, dataTrain.target, cv=5, n_jobs=4)
    
if __name__ == '__main__':
    dataSize = 10000
    dataTrain, dataTest = getTwitterData(size=dataSize)
    train_tfidf, test_tfidf = getTfidfData(dataTrain, dataTest)
    runKNNSimulation(dataTrain, dataTest, train_tfidf, test_tfidf)
    