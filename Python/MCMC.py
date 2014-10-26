

__author__ = 'Varun Nayyar'

import numpy as np
from RobustLikelihoodClass import Likelihood
from MFCCArrayGen import SadCorpus


def weightProp2(currWeights):
    numMixtures = len(currWeights)
    # print numMixtures
    tempWeights = np.zeros(numMixtures + 1)
    tempWeights[1:-1] = np.random.uniform(size=(numMixtures - 1))
    tempWeights[-1] = 1
    return np.diff(np.sort(tempWeights))


def weightProp(currWeights, step=0.01):
    numMixtures = len(currWeights)
    proposedMove = step * np.random.normal(size=numMixtures - 1)
    newWeights = np.zeros(numMixtures)
    newWeights[1:] = currWeights[1:] + proposedMove
    newWeights[0] = 1 - np.sum(newWeights[1:])
    return newWeights


def funTest(numRuns=10000, numMixtures=8):
    Xpoints = SadCorpus()


    # use my flexi object to either use the GPU or CPU depending on what's available
    LLeval = Likelihood(Xpoints, numMixtures)


    #Initialize params
    localMean = Xpoints.mean(0)
    meanRanges = Xpoints.max(0) - Xpoints.min(0)
    meanRanges *= 0.005
    means = np.tile(localMean, (numMixtures, 1)) + meanRanges * np.random.normal(size=(numMixtures, LLeval.dim))

    localVar = Xpoints.var(0)
    diagCovs = np.tile(localVar, (numMixtures, 1)) + 0.01 * localVar * np.random.normal(size=(numMixtures, LLeval.dim))

    weights = np.repeat(1.0 / numMixtures, numMixtures)

    newWeights = np.zeros(numMixtures)

    covIllegal = 0
    acceptNum = 0
    minWeightIllegal = 0
    sumWeightIllegal = 0
    oldLL = LLeval.loglikelihood(means, diagCovs, weights)

    meanList = np.zeros(numRuns)

    print oldLL
    # exit()
    tol = 0.00001


    for i in xrange(numRuns):
        proposalMeans = meanRanges * np.random.normal(size=(numMixtures, LLeval.dim)).astype(np.float32)
        proposalCovs = 0.01 * localVar * np.random.normal(size=(numMixtures, LLeval.dim)).astype(np.float32)
        # proposalweights = 0.1* np.random.normal(size=numMixtures-1).astype(np.float32)

        newMeans = means + proposalMeans

        newCovs = diagCovs + proposalCovs

        # newWeights[1:] = weights[1:] + proposalweights
        #
        # newWeights[0] = 1 - np.sum(newWeights[1:])


        newWeights = weightProp(weights)
        # newWeights = weights

        if (newCovs.min() <= 0):
            covIllegal += 1
            print "{}: Illegal cov proposition: {}".format(i, covIllegal)
            continue

        if newWeights.min() < 0:
            minWeightIllegal += 1
            print "{}: Min Failure: Illegal weight proposition: {}".format(i, minWeightIllegal)
            print newWeights.min(), newWeights.max(), newWeights.sum()
            continue

        if newWeights.sum() < (1.0 - tol) or newWeights.sum()>(1.0+tol):
            sumWeightIllegal += 1
            print "{}: Sum failure: Illegal weight proposition: {}".format(i, sumWeightIllegal)
            print newWeights.min(), newWeights.max(), newWeights.sum()
            continue

        newLL = LLeval.loglikelihood(newMeans, newCovs, newWeights)
        # print newLL

        acceptProb = newLL - oldLL

        if (acceptProb > 0 or acceptProb > np.log(np.random.uniform())):
            means = newMeans
            weights = newWeights
            diagCovs = newCovs
            oldLL = newLL

            acceptNum += 1
            print "{} Accepted!: \t\t{}, {}".format(i, acceptNum, np.exp(acceptProb))


        else:
            print "{} Rejected!: {}".format(i, np.exp(acceptProb))

        meanList[i] = means[0][1]+0


            # break

    print meanList

    print "CovIllegalProps: ", 1.0 * covIllegal / numRuns
    print "WeightIllegalProps: ", 1.0 * minWeightIllegal / numRuns
    print "SumWeightIllegal: ", 1.0 *sumWeightIllegal/numRuns

    print "AcceptedVals: ", 1.0 * acceptNum / numRuns




if __name__ == "__main__":
    # main()
    import sys

    if len(sys.argv) == 2:
        funTest(numRuns=int(sys.argv[1]), numMixtures=8)
    # main(, )
    # We have a input length and numRuns length
    elif len(sys.argv) == 3:
        funTest(numRuns=int(sys.argv[1]), numMixtures=int(sys.argv[2]))
    elif len(sys.argv) == 1:
        # run with default
        funTest()
    else:
        print "Failure in args"