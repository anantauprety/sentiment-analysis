'''
Created on Sep 12, 2015

@author: ananta
'''
from sentiment_data import getTwitterData, getTfidfData, SimpleTimer
from knn import runKNNSimulation
from decision_tree import runDecisionTreeSimulation
from svm import runSVMSimulation
from boosting import runBoosting
from neural_network import runNeuralSimulation

if __name__ == '__main__':
    dataSize = 10000
    dataTrain, dataTest = getTwitterData(size=dataSize)
    train_tfidf, test_tfidf = getTfidfData(dataTrain, dataTest)
    
    func = [runKNNSimulation, runDecisionTreeSimulation, runSVMSimulation, runBoosting]
    for f in func:
        try:
            f()
        except:
            print 'something went wrong'
    print 'running neural network'
    dataSize = 1000
    dataTrain, dataTest = getTwitterData(size=dataSize)
    train_tfidf, test_tfidf = getTfidfData(dataTrain, dataTest)
    try:
        runNeuralSimulation(dataTrain, dataTest, train_tfidf, test_tfidf)
    except:
        print 'threw some error'