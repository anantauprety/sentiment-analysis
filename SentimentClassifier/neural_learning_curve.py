'''
Created on Sep 13, 2015

@author: ananta
'''

if __name__ == '__main__':
    '''
Created on Sep 12, 2015

@author: ananta
'''
from sentiment_data import getTwitterData, getTfidfData, SimpleTimer
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules   import SoftmaxLayer, TanhLayer, SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities           import percentError
from pybrain.structure import RecurrentNetwork, FullConnection

def getDataSetFromTfidf(tfidf, target):
    ds = ClassificationDataSet(test_tfidf.shape[1], nb_classes=2, class_labels=['Negative','Positive'])
    for i in range(tfidf.shape[0]):
        ds.addSample(tfidf[i].toarray()[0], [target[i]])
    ds._convertToOneOfMany()
    return ds
        
def runNeuralLearningCurveSimulation(dataTrain, dataTest, train_tfidf, test_tfidf, outFile):
    print 'running neural learning curve'
    outFile.write('-------------------------------------\n')
    outFile.write('train==> %d, %d \n'%(train_tfidf.shape[0],train_tfidf.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_tfidf.shape[0],test_tfidf.shape[1]))
    
    trainDS = getDataSetFromTfidf(train_tfidf, dataTrain.target)
    testDS = getDataSetFromTfidf(test_tfidf, dataTest.target)
    
    print "Number of training patterns: ", len(trainDS)
    print "Input and output dimensions: ", trainDS.indim, trainDS.outdim
    print "First sample (input, target, class):"
    print len(trainDS['input'][0]), trainDS['target'][0], trainDS['class'][0]
    '''
    with SimpleTimer('time to train', outFile):
        net = buildNetwork(trainDS.indim, trainDS.indim/2, trainDS.indim/4, trainDS.indim/8, trainDS.indim/16, 2, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
        trainer = BackpropTrainer( net, dataset=trainDS, momentum=0.1, verbose=True, weightdecay=0.01, batchlearning=True)
    '''
    net = RecurrentNetwork()
    net.addInputModule(LinearLayer(trainDS.indim, name='in'))
    net.addModule(SigmoidLayer(trainDS.indim/2, name='hidden'))
    net.addModule(SigmoidLayer(trainDS.indim/4, name='hidden2'))
    net.addOutputModule(SoftmaxLayer(2, name='out'))
    net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
    net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))
    net.addRecurrentConnection(FullConnection(net['hidden'], net['hidden'], name='c3'))
    net.addRecurrentConnection(FullConnection(net['hidden2'], net['hidden'], name='c4'))
    net.sortModules()
    trainer = BackpropTrainer( net, dataset=trainDS, momentum=0.01, verbose=True, weightdecay=0.01)
    
    outFile.write('%s \n' % (net.__str__()))
    epochs = 200
    with SimpleTimer('time to train %d epochs' % epochs, outFile):
        for i in range(epochs):
            trainer.trainEpochs(1)
            trnresult = percentError( trainer.testOnClassData(),
                                  trainDS['class'] )
            tstresult = percentError( trainer.testOnClassData(
               dataset=testDS ), testDS['class'] )
    
            print "epoch: %4d" % trainer.totalepochs, \
                  "  train error: %5.2f%%" % trnresult, \
                  "  test error: %5.2f%%" % tstresult
                  
    outFile.write('%5.2f , %5.2f \n' % (100.0-trnresult, 100.0-tstresult))
                  
    predicted = trainer.testOnClassData(dataset=testDS)
    results = predicted == testDS['class'].flatten()
    wrong = []
    for i in range(len(results)):
        if not results[i]:
            wrong.append(i)
    print 'classifier got these wrong:'
    for i in wrong[:10]:
        print dataTest.data[i], dataTest.target[i]
        outFile.write('%s %d \n' % (dataTest.data[i], dataTest.target[i]))
    
if __name__ == '__main__':
    outFile = open('neuralCurveLog.txt','a')
    for size in [100, 200, 500, 700, 1000]:
        dataTrain, dataTest = getTwitterData(size=size)
        train_tfidf, test_tfidf = getTfidfData(dataTrain, dataTest)
        runNeuralLearningCurveSimulation(dataTrain, dataTest, train_tfidf, test_tfidf, outFile)