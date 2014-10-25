__author__ = 'Varun Nayyar'

import numpy as np
from RobustLikelihoodClass import Likelihood
from MFCCArrayGen import SadCorpus


def weightProp(currWeights):
    numMixtures = len(currWeights)
    # print numMixtures
    tempWeights = np.zeros(numMixtures+1)
    tempWeights[1:-1] = np.random.uniform(size=(numMixtures-1))
    tempWeights[-1] = 1
    return np.diff(np.sort(tempWeights))




def funTest(numRuns = 10000, numMixtures = 8):
    Xpoints  = SadCorpus()


    #use my flexi object to either use the GPU or CPU depending on what's available
    LLeval = Likelihood(Xpoints, numMixtures)


    #Initialize params
    localMean = Xpoints.mean(0)
    means = np.tile(localMean, (numMixtures, 1))

    localVar = Xpoints.var(0)
    diagCovs = np.tile(localVar, (numMixtures,1))

    weights = np.repeat(1.0/numMixtures, numMixtures)



    newWeights = np.zeros(numMixtures)


    covIllegal = 0
    acceptNum = 0
    weightIllegal=0
    oldLL = LLeval.loglikelihood(means, diagCovs, weights)

    # print oldLL
    # exit()


    for i in xrange(numRuns):
        proposalMeans = np.random.normal(size=(numMixtures, LLeval.dim)).astype(np.float32)
        proposalCovs = 2 + np.random.normal(size=(numMixtures, LLeval.dim)).astype(np.float32)
        # proposalweights = 0.1* np.random.normal(size=numMixtures-1).astype(np.float32)

        newMeans = means + proposalMeans

        newCovs = diagCovs + proposalCovs

        # newWeights[1:] = weights[1:] + proposalweights
        #
        # newWeights[0] = 1 - np.sum(newWeights[1:])


        newWeights = weightProp(weights)

        if (newCovs.min()<=0):
            covIllegal+=1
            print "{}: Illegal cov proposition: {}".format(i, covIllegal)
            continue

        if (newWeights.min()<0 or newWeights.sum()>1):
            weightIllegal +=1
            print "{}: Illegal weight proposition: {}".format(i, weightIllegal)
            continue

        newLL = LLeval.loglikelihood(newMeans, newCovs, newWeights)

        acceptProb = newLL - oldLL

        if (acceptProb>0 or acceptProb>np.log(np.random.uniform())):
            means = newMeans
            weights = newWeights
            diagCovs = newCovs
            oldLL = newLL

            acceptNum += 1
            print "{} Accepted!: {}".format(i, 1.0*acceptNum/i)
        else:
            print "{} Rejected!: {}".format(i,np.exp(acceptProb))

        # break

    print "CovIllegalProps: ", 1.0*covIllegal/numRuns
    print "WeightIllegalProps: ", 1.0*weightIllegal/numRuns

    print "AcceptedVals: ", 1.0*acceptNum/numRuns


if __name__ == "__main__":
    # main()
    funTest()