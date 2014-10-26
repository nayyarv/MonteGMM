__author__ = 'Varun Nayyar'

import numpy as np
from matplotlib import pyplot as plt
from RobustLikelihoodClass import Likelihood


def main():
    pass


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
    main()