'''
Created on Sep 12, 2015

# just trying to change something
@author: ananta
'''

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sentiment_data import getTwitterData, getTfidfData, SimpleTimer
from plot_learning_curve import plot_learning_curve

def runSVMSimulation(dataTrain, dataTest, train_tfidf, test_tfidf):
    outFile = open('svmLog.txt','a')
    
    outFile.write('train==> %d, %d \n'%(train_tfidf.shape[0],train_tfidf.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_tfidf.shape[0],test_tfidf.shape[1]))
    with SimpleTimer('time to train', outFile):
#         clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=30, random_state=42)
        clf = LinearSVC(C=1.0)
        clf.fit(train_tfidf, dataTrain.target)
    
    baseScore = clf.score(test_tfidf, dataTest.target)
    baseIter = 5
    print 'baseline score %.3f base n_neighbors %d' % (baseScore, baseIter)
    outFile.write('baseline score %.3f base height %d \n' % (baseScore, baseIter))
    
    res = []
    with SimpleTimer('number of iter', outFile):
        for itr in range(1):
            print 'training for neighbors %d' % itr
#             clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=itr, random_state=42)
            clf = LinearSVC(loss='squared_hinge', C=1.0)
            clf.fit(train_tfidf, dataTrain.target)
            score = clf.score(test_tfidf, dataTest.target)
            res.append((score, itr))
            outFile.write('%d %.3f \n' % (itr, score))
    res = sorted(res, key=lambda x:x[0], reverse=True)
    print res[:5]
    bestItr = res[0][1]
    print ('best number of iter is %d' % bestItr) 
    bestClf = LinearSVC(loss='squared_hinge', C=1.0)
    bestClf.fit(train_tfidf, dataTrain.target)
    predicted = bestClf.predict(test_tfidf)
    
    results = predicted == dataTest.target
    res = []
    for i in range(len(results)):
        if not results[i]:
            res.append(i)
    print 'classifier got these wrong:'
    for i in res[:10]:
        print dataTest.data[i], dataTest.target[i]
        outFile.write('%s %d \n' % (dataTest.data[i], dataTest.target[i]))
    '''
    bestClf = LinearSVC(loss='squared_hinge', C=1.0)
    plot_learning_curve(bestClf, 'svm with %d iter' % bestItr, train_tfidf, dataTrain.target, cv=5, n_jobs=4)
    
    bestClf = LinearSVC(loss='hinge', C=1.0)
    plot_learning_curve(bestClf, 'svm hinge with %d iter' % bestItr, train_tfidf, dataTrain.target, cv=5, n_jobs=4)
    '''
if __name__ == '__main__':
    dataSize = 50000
    dataTrain, dataTest = getTwitterData(size=dataSize)
    train_tfidf, test_tfidf = getTfidfData(dataTrain, dataTest)
    runSVMSimulation(dataTrain, dataTest, train_tfidf, test_tfidf)