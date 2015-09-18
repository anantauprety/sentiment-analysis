'''
Created on Sep 12, 2015

@author: ananta
'''

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sentiment_data import getTwitterData, getTfidfData, SimpleTimer, getPickeledData, outputScores
from plot_learning_curve import plot_learning_curve
from sklearn.metrics import confusion_matrix

def runSVMSimulation(dataTrain, dataTest, holdOut, train_tfidf, test_tfidf, hold_tfidf):
    kernel = 'poly'
    penalty = 1.0
    outFile = open('svmLog%s.txt' % kernel,'a')
    degree = 3
    outFile.write('train==> %d, %d \n'%(train_tfidf.shape[0],train_tfidf.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_tfidf.shape[0],test_tfidf.shape[1]))
    
    with SimpleTimer('time to train', outFile):
#         clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=30, random_state=42)
        clf = SVC(kernel=kernel, C=penalty, degree=degree)
        clf.fit(train_tfidf, dataTrain.target)
    
    baseScore = clf.score(test_tfidf, dataTest.target)
    baseIter = 5
    print 'baseline score %.3f penalty %d' % (baseScore, baseIter)
    outFile.write('baseline score %.3f base height %d \n' % (baseScore, baseIter))
    
    res = []
    with SimpleTimer('number of iter', outFile):
        for pen in [1,2,3,4,5]:
            print 'training for peanalty %f' % pen
#             clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=itr, random_state=42)
            clf = SVC(kernel=kernel, C=1.0, degree=pen)
            clf.fit(train_tfidf, dataTrain.target)
            score = clf.score(hold_tfidf, holdOut.target)
            res.append((score, pen))
            outFile.write('%.3f %.3f \n' % (pen, score))
            
    res = sorted(res, key=lambda x:x[0], reverse=True)
    print res[:5]
    bestPen = res[0][1]
    print ('best number of iter is %.3f' % bestPen) 
    bestClf = SVC(kernel=kernel, C=1.0, degree=bestPen)
    bestClf.fit(train_tfidf, dataTrain.target)
    
    train_predict = bestClf.predict(train_tfidf)
    predicted = bestClf.predict(test_tfidf)
    
    print 'testing score'
    outFile.write('testing score')
    outputScores(dataTest.target, predicted, outFile)
    print 'training score'
    outFile.write('testing score')
    outputScores(dataTrain.target, train_predict, outFile)
    
    
    
    results = predicted == dataTest.target
    res = []
    for i in range(len(results)):
        if not results[i]:
            res.append(i)
    print 'classifier got these wrong:'
    for i in res[:10]:
        print dataTest.data[i], dataTest.target[i]
        outFile.write('%s %d \n' % (dataTest.data[i], dataTest.target[i]))
    
    plot_learning_curve(bestClf, 'svm with %s kernel & degree %.3f' % (kernel, bestPen), train_tfidf, dataTrain.target, cv=5, n_jobs=4)
    '''
    bestClf = LinearSVC(loss='hinge', C=1.0)
    plot_learning_curve(bestClf, 'svm hinge with %d iter' % bestItr, train_tfidf, dataTrain.target, cv=5, n_jobs=4)
    '''
if __name__ == '__main__':
    dataSize = 10000
#     dataTrain, dataTest = getTwitterData(size=dataSize)
#     train_tfidf, test_tfidf = getTfidfData(dataTrain, dataTest)
    dataTrain, dataTest, holdOut, train_tfidf, test_tfidf, hold_tfidf = getPickeledData()
    runSVMSimulation(dataTrain, dataTest, holdOut, train_tfidf, test_tfidf, hold_tfidf)
    
