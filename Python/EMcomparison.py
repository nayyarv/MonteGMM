__author__ = 'Varun Nayyar'

import numpy as np
from MFCCArrayGen import getCorpus, getIndiviudalData, emotions, speakers
from sklearn.mixture import GMM
from sklearn.metrics import confusion_matrix
from RobustLikelihoodClass import Likelihood



def main(numMixtures = 8):
    Xpoints = getCorpus("Sad")
    numPoints, dim = Xpoints.shape

    llEval = Likelihood(Xpoints, numMixtures)

    Model = GMM(8, n_iter=1000)
    Model.fit(Xpoints)

    print np.sort(Model.weights_)

    print llEval.loglikelihood(Model.means_, Model.covars_, Model.weights_)
    print np.sum(Model.score(Xpoints))

    ##above two are the same!! Yay


def main2(numMixtures = 8, speakerIndex = 6):

    modelDict = {}

    y_test = []
    y_pred = []

    for emotion in emotions:
        Xpoints = getCorpus(emotion, speakers[speakerIndex])
        modelID = emotion
        modelDict[modelID] = GMM(8, n_iter=10000,)

        modelDict[modelID].fit(Xpoints)

    #testing
    for testEmotion in emotions:
        testCorpus = getIndiviudalData(testEmotion, speakers[speakerIndex])

        for utterance in testCorpus:
            LLEmotion = ""
            maxEmotionVal = -1e10

            #Classify it!
            for modelEmotion in emotions:
                llVal = np.sum(modelDict[modelEmotion].score(utterance))
                #gives me the likelihood of being part of that
                # print "Actual Emotion: {}, TestForEmotion: {}, value:{}".format(testEmotion, modelEmotion, llVal)

                if llVal > maxEmotionVal:
                    LLEmotion = modelEmotion
                    maxEmotionVal = llVal

            # print "Given: {}, Chosen Emotion: {}".format(testEmotion, LLEmotion)
            y_test.append(testEmotion)
            y_pred.append(LLEmotion)

        print ""
    cm = confusion_matrix(y_test, y_pred, labels = emotions)
    # print emotions
    # print ""
    return cm



def fullTest():
    TotalCM = np.zeros((7, 5,5))

    for i in xrange(len(speakers)):
        TotalCM[i] = main2( speakerIndex=i)
        print speakers[i]
        print emotions
        print TotalCM[i]
        print "Normalise Totals :", TotalCM[i].sum(1)

    print emotions
    print TotalCM.sum(0)
    print "Normalise Totals :", TotalCM.sum(0).sum(1)



if __name__ == "__main__":
    fullTest()