__author__ = 'Varun Nayyar'

import numpy as np
from matplotlib import pyplot as plt


def main():
    pass

def adaptive():
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
            print "increasing menStep"
        elif meanAccProb < 0.25:
            localMean/=exp_deltan
            print "reducing meanStep"

        if covAccProb > 0.35:
            localVar *= exp_deltan
            print "increasing covStep"
        elif covAccProb < 0.25:
            localVar /=exp_deltan
            print "reducing covStep"

            #otherwise

        if weightAccProb > 0.35:
            weightStep *= exp_deltan
            print "increasing weightStep"
        elif weightAccProb < 0.25:
            weightStep /= exp_deltan
            print "reducing weightStep"

        meanBatchAcceptance[:] = 0
        covBatchAcceptance[:] = 0
        weightBatchAcceptance = 0


def adaptMeans():

    for mixture in xrange(LLeval.numMixtures):
        meansStorage[i-1]
        newMeans = means+0

        newMeans[mixture] = means[mixture] + \
                             np.random.multivariate_normal(size = LLeval.dim).astype(np.float32)

        newLL = LLeval.loglikelihood(newMeans, diagCovs, weights)

        acceptProb = newLL - oldLL

        if acceptProb > 0 or acceptProb > np.log(np.random.uniform()):

            means[mixture] = newMeans[mixture]

            oldLL = newLL

            overallMeanAcceptance[mixture]+=1





if __name__ == "__main__":
    main()