

__author__ = 'Varun Nayyar'

import numpy as np
from RobustLikelihoodClass import Likelihood
from MFCCArrayGen import SadCorpus
from scipy.stats import norm

def weightProp2(currWeights):
    numMixtures = len(currWeights)
    # print numMixtures
    tempWeights = np.zeros(numMixtures + 1)
    tempWeights[1:-1] = np.random.uniform(size=(numMixtures - 1))
    tempWeights[-1] = 1
    return np.diff(np.sort(tempWeights))


def weightAcceptanceMod(newWeights, currWeights, step = 0.01):
    if (currWeights[1:].min()>0.03 or newWeights[1:].min()>0.3): return 0
    currWeights = currWeights[1:]/step
    oldCdf = norm.cdf(currWeights)
    newWeights = newWeights[1:]/step
    newCdf = norm.cdf(newWeights)

    # print oldCdf, newCdf

    AcceptMod = np.sum(np.log(oldCdf)) - np.sum(np.log(newCdf))

    print "AcceptMod: ", AcceptMod
    return AcceptMod

def weighPropPositive(currWeights, step = 0.01):

    numMixtures = len(currWeights)
    newWeights = np.zeros(numMixtures)

    while newWeights.min()<0 or newWeights.max() ==0:
        proposedMove = step * np.random.normal(size=numMixtures - 1)
        newWeights[:] = 0
        newWeights[1:] = currWeights[1:] + proposedMove
        newWeights[0] = 1 - np.sum(newWeights[1:])

    return newWeights, weightAcceptanceMod(newWeights, currWeights, step)


def weightPropOld(currWeights, step=0.01):
    numMixtures = len(currWeights)
    proposedMove = step * np.random.normal(size=numMixtures - 1)
    newWeights = np.zeros(numMixtures)
    newWeights[1:] = currWeights[1:] + proposedMove
    newWeights[0] = 1 - np.sum(newWeights[1:])
    return newWeights






def funTest(numRuns=10000, numMixtures=4):
    Xpoints = np.vstack(SadCorpus())


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
    # weights = np.array([1]+[0]*(numMixtures-1))
    newWeights = np.zeros(numMixtures)

    covIllegal = 0
    acceptNum = 0
    minWeightIllegal = 0
    sumWeightIllegal = 0
    oldLL = LLeval.loglikelihood(means, diagCovs, weights)

    meanList = np.zeros(numRuns)

    weightList = np.zeros((numRuns,numMixtures))

    print oldLL
    # exit()
    tol = 0.00001

    meanBatchAcceptance = np.zeros(numMixtures)
    covBatchAcceptance = np.zeros(numMixtures)
    weightBatchAcceptance = 0

    overallMeanAcceptance = np.zeros(numMixtures)
    overallCovAcceptance = np.zeros(numMixtures)
    overallWeightAcceptance = 0


    localMean*=0.02
    localMean = np.abs(localMean)
    print "LocalMean: ", localMean
    # print np.log(localMean)

    localVar*=0.02
    localVar = np.abs(localVar)
    print "LocalVars: ", localVar
    # print np.log(localVar)

    weightStep = 0.005

    # exit()


    for i in xrange(numRuns):
        # proposalMeans = 0.02 * localMean * np.random.normal(size=(numMixtures, LLeval.dim)).astype(np.float32)

        for mixture in xrange(LLeval.numMixtures):
            newMeans = means+0
            #copy, not point
            #Reinitialize
            newMeans[mixture] = means[mixture] + \
                                 localMean * np.random.normal(size = LLeval.dim).astype(np.float32)

            newLL = LLeval.loglikelihood(newMeans, diagCovs, weights)

            acceptProb = newLL - oldLL

            if acceptProb > 0 or acceptProb > np.log(np.random.uniform()):
                #we have acceptance!
                means[mixture] = newMeans[mixture]
                print "\t\t{}: Mean of mixture {} accepted, {}".format(i, mixture, acceptProb)
                oldLL = newLL
                meanBatchAcceptance[mixture]+=1
                overallMeanAcceptance[mixture]+=1
            else:
                print "{}: Mean of mixture {} Rejected, {}".format(i, mixture, acceptProb)


        proposalCovs = np.random.normal(size=(numMixtures, LLeval.dim)).astype(np.float32)

        for mixture in xrange(LLeval.numMixtures):
            newCovs = diagCovs+0 #reinitialize, copy not point
            newCovs[mixture] = diagCovs[mixture] + localVar * np.random.normal(size=LLeval.dim).astype(np.float32)

            if newCovs.min() <= 0.01:
                covIllegal += 1
                print "{}: Illegal cov of mixture: {} proposition: {}".format(i,mixture, covIllegal)
                continue

            newLL = LLeval.loglikelihood(means, newCovs, weights)
            acceptProb = newLL - oldLL

            if acceptProb > 0 or acceptProb > np.log(np.random.uniform()):
                #we have acceptance!
                diagCovs[mixture] = newCovs[mixture]
                print "\t\t{}, Cov of mixture {} accepted, {}".format(i, mixture, acceptProb)
                oldLL = newLL
                covBatchAcceptance[mixture]+=1
                overallCovAcceptance[mixture]+=1
            else:
                print "{}: Cov of mixture {} Rejected, {}".format(i, mixture, acceptProb)


        newWeights, weightAcceptMod = weighPropPositive(weights, step = weightStep)
        # newWeights = weights


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

        newLL = LLeval.loglikelihood(means, diagCovs, newWeights)
        # print newLL

        acceptProb = newLL - oldLL + weightAcceptMod

        if acceptProb > 0 or acceptProb > np.log(np.random.uniform()):
            weights = newWeights
            oldLL = newLL
            print "\t\t{}: Weight Accepted!: {}, {}".format(i, acceptNum, acceptProb)
            weightBatchAcceptance+=1
            overallWeightAcceptance+=1
        else:
            print "{}: Weight Rejected!: {}, {}".format(i, acceptNum, acceptProb)

        meanList[i] = means[0][1]+0
        weightList[i] = weights+0

        if (i-1)%50 ==0:
            #Update Step sizes
            n = i/50
            delta_n = min(0.01, 1/np.sqrt(n))
            exp_deltan = np.exp(delta_n)
            #acceptance probabilities
            meanAccProb = np.mean(meanBatchAcceptance/(i*1.0))
            covAccProb = np.mean(covBatchAcceptance/(i*1.0))
            weightAccProb = weightBatchAcceptance/(i*1.0)

            print "Acceptance rate for batch {} is:".format(n)
            print "Means: ", meanAccProb
            print "Covs: ", covAccProb
            print "Weights: ", weightAccProb



            if meanAccProb > 0.35: # too high
                localMean*=exp_deltan
            elif meanAccProb < 0.25:
                localMean/=exp_deltan

            if covAccProb > 0.35:
                localVar *= exp_deltan
            elif covAccProb < 0.25:
                localVar /=exp_deltan

                #otherwise

            if weightAccProb > 0.35:
                weightStep *= exp_deltan
            elif weightAccProb < 0.25:
                weightStep /= exp_deltan

            meanBatchAcceptance[:] = 0
            covBatchAcceptance[:] = 0
            weightBatchAcceptance = 0

        # break

    print meanList

    print weightList.mean(0), weightList.var(0)
    print np.sort(weightList[-1])


    print "CovIllegalProps: ", 1.0 * covIllegal / numRuns
    print "WeightIllegalProps: ", 1.0 * minWeightIllegal / numRuns
    print "SumWeightIllegal: ", 1.0 *sumWeightIllegal/numRuns


    print "Mean Acceptance: ",meanBatchAcceptance
    print "Cov Acceptance: ", covBatchAcceptance
    print "Weight Acceptance: ", weightBatchAcceptance




if __name__ == "__main__":
    # main()
    import sys

    if len(sys.argv) == 2:
        funTest(numRuns=int(sys.argv[1]))
    # main(, )
    # We have a input length and numRuns length
    elif len(sys.argv) == 3:
        funTest(numRuns=int(sys.argv[1]), numMixtures=int(sys.argv[2]))
    elif len(sys.argv) == 1:
        # run with default
        funTest()
    else:
        print "Failure in args"