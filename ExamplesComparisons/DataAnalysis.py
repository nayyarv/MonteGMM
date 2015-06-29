import cPickle
from matplotlib import pyplot as plt
import numpy as np

print "Starting"

numPoints = 64

with open("../Data/Mean2,3_{}pts_1dim_MCMCRes800000.txt".format(numPoints)) as f:
    samples = cPickle.load(f)

print "Finished UnPickling"

burnIn = 1000
endPoint = 100000
lag = 100

acorrOnly = False

if acorrOnly:

    # plt.figure(tight_layout=True)
    plt.subplot(211)
    plt.title("Autocorrelation of samples")
    plt.ylabel("Correlation")
    plt.xlim((-1, 101))
    plt.acorr(samples.T[0][burnIn:endPoint] - np.mean(samples.T[0][burnIn:endPoint]), maxlags=100)
    # pri/nt samples.T[0][burnIn:]

    plt.subplot(212)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.xlim((-1, 101))
    plt.acorr(samples.T[1][burnIn:endPoint] - np.mean(samples.T[1][burnIn:endPoint]), maxlags=100)
    # print samples.T[1][burnIn:]
    plt.show()

else:
    from EMComparisonToy import EMSpread
    EMmeans = EMSpread(numPoints)
    plt.figure(tight_layout=True)
    plt.title("log $p(\\theta|x)$ and 100 EM estimates, {} points ".format(numPoints))
    plt.xlabel("$\mu_1$")
    plt.ylabel("$\mu_2$")
    plt.hexbin(samples.T[0][burnIn::lag], samples.T[1][burnIn::lag], bins='log')
    # plt.plot(EMmeans.T[0], EMmeans.T[1], 'x')

    plt.show()