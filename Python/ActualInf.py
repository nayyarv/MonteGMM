
import numpy as np
from MFCCArrayGen import getCorpus, getIndiviudalData, emotions, speakers

# from sklearn.metrics import confusion_matrix
from RobustLikelihoodClass import Likelihood

import sys, os

def BayesProb(utterance, numMixtures, means, diagCovs, weights):
    """

    Given the MCMC values from a run, calculate probability of belonging to that class

    :param utterance: np.array of shape [size][dim]
    :param numMixtures:
    :param means: np.array [numMCMCRuns][numMixtures][dim]
    :param diagCovs: np.array [numMCMCRuns][numMixtures][dim]
    :param weights: np.array [numMCMCRuns][numMixtures]
    :return:
    """
    sys.stdout = open(os.devnull, "w")
    llEval = Likelihood(utterance, numMixtures=8)
    sys.stdout = sys.__stdout__

    prob = 0

    for i in xrange(means.shape[0]):
        prob+= llEval.loglikelihood(means[i], diagCovs[i], weights[i])
    # print prob/means.shape[0]

    return prob/means.shape[0]

def main():
    y_test = []
    y_pred = []
    speakerIndex = 0
    numMixtures = 8


    filename = "../SpeechMCMC/{}-CC1.txt".format(sys.argv[1])
    import cPickle

    print filename

    with open(filename) as f:
        MCMCmeans, MCMCcovs, MCMCweights = cPickle.load(f)

    for emotion in emotions:
        testCorpus = getIndiviudalData(emotion, speakers[speakerIndex])

        print "Actual Emotion: {}".format(emotion)

        for utterance in testCorpus:
            print "Likelihood it's {}".format(sys.argv[1]), BayesProb(utterance, 8, MCMCmeans, MCMCcovs, MCMCweights) 

if __name__ == '__main__':
    main()