__author__ = 'Varun Nayyar'

import numpy as np
from MFCCArrayGen import emotions, speakers, getIndiviudalData, getCorpus
from MCMC import MCMCRun
from RobustLikelihoodClass import Likelihood
from emailScripy import alertMe


def main2(numRuns = 100000, numMixtures = 8, speakerIndex = 6):
    import time

    

    for emotion in emotions:
        start = time.ctime()

        Xpoints = getCorpus(emotion, speakers[speakerIndex])


        message = MCMCRun(Xpoints, emotion+"-"+speakers[speakerIndex], numRuns, numMixtures)

        message += "Start time: {}\nEnd Time: {}\n".format(start, time.ctime())

        message += "\nNumRuns: {}, numMixtures:{}\n ".format(numRuns, numMixtures)

        message += "\nEmotion: {}, speaker:{}\n".format(emotion, speakers[speakerIndex])

    alertMe(message)



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

    llEval = Likelihood(utterance, numMixtures=numMixtures)

    prob = 0

    for i in xrange(means.shape[0]):
        prob+= np.exp(llEval.loglikelihood(means[i], diagCovs[i], weights[i]))
    print prob/means.shape[0]

    return prob/means.shape[0]


if __name__ == "__main__":
    for i in xrange(len(speakers)):
        main2(numMixtures=8, speakerIndex=i)