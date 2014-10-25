# Monte Carlo



# import pycuda.driver as drv
# import pycuda.autoinit
# from pycuda.compiler import SourceModule

# with open("KernelV2.cu") as f:
# 	mod = SourceModule(f.read())

# likelihoodKernel = mod.get_function("likelihoodKernel")
# likelihoodKernel.prepare('PPPPiiiP')

import numpy as np
from scipy.stats import norm
from emailScripy import alertMe
import cPickle
import time


dim = np.int32(1)
numMixtures = np.int32(2)


def main(inputDataLen=16, numRuns=80):
    numPoints = inputDataLen / 2  #Number of each

    startTime = time.ctime()

    #Generated data!!
    with open("../FixedDataSet/Mean2,3;{}pts;1dim.txt".format(inputDataLen)) as f:
        Xpoints = cPickle.load(f)


    # plt.hist(Xpoints)
    # plt.show()

    weights = np.ones(numMixtures) / (1.0 * numMixtures)
    weights = weights.astype(np.float32)
    # means = np.arange(numMixtures*dim).reshape((numMixtures,dim))
    # means=means.astype(np.float32)

    #Initial mean
    # means = np.random.normal(loc = 1.5, scale = 1, size=(numMixtures,dim)).astype(np.float32)


    means = np.array([[2], [3]]).astype(np.float32)

    oldLL = -10000
    newLL = 0
    ll = np.zeros(numPoints * numMixtures)
    samples = np.zeros((numRuns, 2))
    diagVal = 1.5
    acceptNum = 0
    currentPos = 0
    for k in xrange(numRuns):
        if k % 100 == 0:
            print "At ", k, " iterations"

        if currentPos > 1:
            diagVal = 0.2
        else:
            diagVal = 1.5

        proposalFunc = np.random.multivariate_normal(mean=[0, 0], cov=np.diag([diagVal, diagVal]))
        ll[:] = 0

        for i in xrange(numPoints * numMixtures):
            for mixes in xrange(numMixtures):
                multiVal = 1.0
                for d in xrange(dim):
                    x = Xpoints[i][d]
                    m = means[mixes][d] + proposalFunc[mixes]

                    c = 1

                    # multiVal*= 1/np.sqrt(2*np.pi*c)*np.exp(-1.0/2 * ((x-m)**2)/c)
                    multiVal *= norm.pdf(x, loc=m, scale=np.sqrt(c))

                ll[i] += multiVal / weights[mixes]

        newLL = np.sum(np.log(ll))
        acceptProb = np.exp(newLL - oldLL)

        if ( acceptProb >= 1 or acceptProb > np.random.uniform()):
            means.T[0] += proposalFunc
            oldLL = newLL
            acceptNum += 1
            currentPos = 0
        # print k, " A ", means.T[0], proposalFunc, newLL, acceptProb
        else:
            currentPos += 1
        # print k, "N ", means.T[0] + proposalFunc, proposalFunc, newLL, acceptProb

        samples[k] = (means.T[0] + 0)

    print acceptNum

    print "Pickling"

    with open("../Data/Mean2,3;{}pts;1dim;MCMCRes{}.txt".format(inputDataLen, numRuns), 'w') as f:
        cPickle.dump(samples, f)

    endTime = time.ctime()
    print "Alerting Varun"

    if numRuns >= 8000:
        import sys

        alertMe("\n{}\nStart: {}\nEnd: {}\nAcceptProb: {}\n".format(sys.argv, startTime, endTime,
                                                                    (1.0 * acceptNum) / numRuns))


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 3:
        main(int(sys.argv[1]), int(sys.argv[2]))
    #We have a input length and numRuns length
    elif len(sys.argv) == 1:
        #run with default
        main()
    else:
        print "Failure"
        # main()