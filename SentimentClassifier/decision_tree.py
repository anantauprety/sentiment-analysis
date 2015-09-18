'''
Created on Sep 12, 2015

@author: ananta
'''

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import numpy
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import logging
from sentiment_data import getTwitterData, getTfidfData, SimpleTimer, getPickeledData, outputScores
from sklearn import tree
from sklearn.learning_curve import learning_curve
from sklearn.metrics import accuracy_score
from plot_learning_curve import plot_learning_curve
logger = logging.getLogger('training')

def printPdf(clf, dataTrain):
    from sklearn.externals.six import StringIO
    import pydot
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('sentiment.pdf')
    print dataTrain.data[0]

def runDecisionTreeSimulation(dataTrain, dataTest, dataHold, train_tfidf, test_tfidf, hold_tfidf):
    print 'running decision tree'
    outFile = open('decisionTreeLog.txt','a')

    outFile.write('train==> %d, %d \n'%(train_tfidf.shape[0],train_tfidf.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_tfidf.shape[0],test_tfidf.shape[1]))
    with SimpleTimer('time to train', outFile):
        clf = DecisionTreeClassifier().fit(train_tfidf, dataTrain.target)
    
    baseScore = clf.score(test_tfidf, dataTest.target)
    initHeight = clf.tree_.max_depth
    print 'baseline score %.3f base height %d' % (baseScore, initHeight)
    outFile.write('baseline score %.3f base height %d \n' % (baseScore, initHeight))
    
    
    res = []
    with SimpleTimer('time to prune', outFile):
        for height in range(initHeight, 40, -25):
#             print 'training for height %d' % height
            clf = DecisionTreeClassifier(max_depth=height).fit(train_tfidf, dataTrain.target)
            score = clf.score(hold_tfidf, dataHold.target)
            res.append((score, height))
            outFile.write('%d %.3f \n' % (height, score))
    res = sorted(res, key=lambda x:x[0], reverse=True)
    print res[:5]
    
    bestDepth = res[0][1]
    print ('best height is %d' % bestDepth)
    outFile.write('best depth is %d  and score is %.3f \n' % (bestDepth, res[0][0]))
        
    bestClf = DecisionTreeClassifier(max_depth=bestDepth)
    bestClf.fit(train_tfidf, dataTrain.target)
    
    predicted = bestClf.predict(test_tfidf)
    
    train_predict = bestClf.predict(train_tfidf)
    
    print 'testing score'
    outFile.write('testing score')
    outputScores(dataTest.target, predicted, outFile)
    print 'training score'
    outFile.write('testing score')
    outputScores(dataTrain.target, train_predict, outFile)
    
    results = predicted == dataTest.target
    wrong = []
    for i in range(len(results)):
        if not results[i]:
            wrong.append(i)
    print 'classifier got these wrong:'
    for i in wrong[:10]:
        print dataTest.data[i], dataTest.target[i]
        outFile.write('%s %d \n' % (dataTest.data[i], dataTest.target[i]))
    plot_learning_curve(bestClf, 'decision tree after pruning from %d to %d depth' % (initHeight, bestDepth), train_tfidf, dataTrain.target, cv=5, n_jobs=4)
    
if __name__ == '__main__':
    dataSize = 10000
#     dataTrain, dataTest = getTwitterData(size=dataSize)
#     train_tfidf, test_tfidf = getTfidfData(dataTrain, dataTest)
    dataTrain, dataTest, holdOut, train_tfidf, test_tfidf, hold_tfidf = getPickeledData()
    runDecisionTreeSimulation(dataTrain, dataTest, holdOut, train_tfidf, test_tfidf, hold_tfidf)